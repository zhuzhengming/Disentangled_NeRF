import numpy as np
import torch
import torch.nn as nn
import json
from DATA import DATA_loader
from UTILS import get_rays, sample_from_rays, volume_rendering, image_float_to_uint8
from MODEL import ConNeRF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import math
import time

class Trainer(object):
    def __init__(self, save_dir, gpu,checkpoints_path, jsonfile='config.json', batch_size=1024,
                 check_iter = 10000):
        super().__init__()

        # load Hyperparameters
        config_path = os.path.join('config_file', jsonfile)
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.device = torch.device('cuda:' + str(gpu))
        self.batch_size = batch_size
        self.check_iter = check_iter
        self.saver(save_dir)
        self.niter, self.nepoch = 0, 0
        self.checkpoints_path = checkpoints_path

        #load info from config_file
        self.lr1 = self.config['lr_schedule'][0]
        self.lr2 = self.config['lr_schedule'][1]
        self.loss_reg_coef = self.config['loss_reg_coef']
        self.near = self.config['near']
        self.far = self.config['far']
        self.N_samples = self.config['N_samples']
        self.check_points = self.config['check_points']
        self.model = ConNeRF(**self.config['net_hyperparams']).to(self.device)

        self.trainer_dataloader(num_instances_per_obj=1, crop_img=False)
        self.genarate_codes()

    def trainer_dataloader(self, num_instances_per_obj, crop_img):
        # num_instances_per_obj : how many images we choose from objects
        # crop_img : whether to crop image or not
        data_dir = self.config['data']['data_dir']
        splits = self.config['data']['splits']
        DATA = DATA_loader(data_dir = data_dir, splits = splits,
                           num_intances_per_obj=num_instances_per_obj,
                                mode = 'train', crop = crop_img)
        self.dataloader = DataLoader(DATA, batch_size=1, num_workers=4)

    def genarate_codes(self):
        embdim = self.config['net_hyperparams']['latent_dim']
        d = len(self.dataloader)

        self.Zs_codes = nn.Embedding(d, embdim)
        self.Zt_codes = nn.Embedding(d, embdim)

        # init latent code:
        self.Zs_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))
        self.Zt_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))

        self.Zs_codes = self.Zs_codes.to(self.device)
        self.Zt_codes = self.Zt_codes.to(self.device)

    def saver(self, save_dir):
        self.save_dir = save_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(os.path.join(self.save_dir,'runs'))

        #tensorboard record tools
        self.writer = SummaryWriter(os.path.join(self.save_dir,'runs'))

    def save_model(self, iters = None):
        #save the model
        save_dict = {'model_params': self.model.state_dict(),
                     'Zs_code_params': self.Zs_codes.state_dict(),
                     'Zt_code_params': self.Zt_codes.state_dict(),
                     'niter': self.niter,
                     'nepoch': self.nepoch
                     }
        if iters != None:
            # record after training for specific number epochs
            torch.save(save_dict, os.path.join(self.save_dir, str(iters) + '.pth'))
        #keep saving current model
        torch.save(save_dict, os.path.join(self.save_dir, 'models.pth'))

    def optimizers(self):
        num_model = self.niter // self.lr1['interval']
        num_latent = self.niter // self.lr2['interval']
        lr1 = self.lr1['lr'] * 2 ** (-num_model)
        lr2 = self.lr2['lr'] * 2 ** (-num_latent)
        self.optimizer = torch.optim.AdamW([
            {'params':self.model.parameters(), 'lr': lr1},
            {'params':self.Zs_codes.parameters(), 'lr': lr2},
            {'params':self.Zt_codes.parameters(), 'lr': lr2}
        ])

    def training_single_epoch(self, num_instances_per_obj, num_iters, crop_img = True):
        self.trainer_dataloader(num_instances_per_obj, crop_img=crop_img)
        self.optimizers()
        #per object
        for obj in self.dataloader:
            if self.niter < num_iters:
                focal, H, W, imgs, poses, instances, obj_idx = obj
                obj_idx = obj_idx.to(self.device)

                #per image
                self.optimizer.zero_grad()
                for k in range(num_instances_per_obj):
                    # print(k, num_instances_per_obj, poses[0, k].shape, imgs.shape, 'k')
                    t1 = time.time()
                    self.optimizer.zero_grad()

                    # Sampling
                    rays_o, viewdir = get_rays(H.item(), W.item(), focal, poses[0, k])
                    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, self.near, self.far, self.N_samples)

                    loss_per_img, generated_img = [], []
                    # start, stop, step = batch_size
                    for i in range(0, xyz.shape[0], self.batch_size):
                        Zs_codes, Zt_codes = self.Zs_codes(obj_idx), self.Zt_codes(obj_idx)
                        sigma, rgb = self.model(xyz[i:i+self.batch_size].to(self.device),
                                                viewdir[i:i+self.batch_size].to(self.device),
                                                Zs_codes,
                                                Zt_codes
                                                )

                        rgb_rays, _ = volume_rendering(sigma, rgb, z_vals.to(self.device))

                        #LOSS calculation
                        #Euclidean distance
                        loss_l2 = torch.mean((rgb_rays - imgs[0, k, i:i+self.batch_size].type_as(rgb_rays))**2)

                        #why?  loss
                        if i == 0:
                            latent_loss = torch.norm(Zs_codes, dim=-1) + torch.norm(Zt_codes, dim=-1)
                            loss_reg = self.loss_reg_coef * torch.mean(latent_loss)
                            # LOSS = ||C'(r) - C(r)|| + (1/v^2)(||Zs^2|| + ||Zt^2||)
                            loss = loss_l2 + loss_reg
                        else:
                            loss = loss_l2

                        loss.backward()

                        #record
                        loss_per_img.append(loss_l2.item())
                        generated_img.append(rgb_rays)
                #update parameters
                self.optimizer.step()

                # log
                self.logger(np.mean(loss_per_img),time.time()-t1,loss_reg,loss_l2,loss,obj_idx)

                #save image
                if self.niter % self.check_iter == 0:
                    generated_img = torch.cat(generated_img)
                    generated_img = generated_img.reshape(H,W,3)
                    gtimg = imgs[0,-1].reshape(H,W,3)
                    self.image_log(generated_img, gtimg, obj_idx)

                # record nums of check_points
                if self.niter % self.check_iter == 0:
                    self.save_model(self.niter)
                self.niter += 1
                print("niter:",self.niter)

    def training(self, iters, num_instances_per_obj = 1):
        if os.path.exists(os.path.join(self.checkpoints_path,"models.pth")):
            checkpoint = torch.load(os.path.join(self.checkpoints_path,"models.pth"))
            self.model.load_state_dict(checkpoint['model_params'])
            self.model = self.model.to(self.device)
            self.Zs_codes.load_state_dict(checkpoint['Zs_code_params'])
            self.Zt_codes.load_state_dict(checkpoint['Zt_code_params'])
            self.niter = checkpoint['niter']
            self.nepoch = checkpoint['nepoch']
        while self.niter < iters:
            self.training_single_epoch(num_instances_per_obj, iters, True)
            self.save_model()
            self.nepoch += 1
            print("nepoch:",self.nepoch)

    def logger(self, loss_per_img, time_spent, loss_reg, loss_l2, loss, obj_idx):
        #Peak Signal-to-Noise Ratio: evaluate the difference between GT and generated image
        psnr = -10*np.log(loss_per_img) / np.log(10)
        self.writer.add_scalar('panr/train', psnr, self.niter, obj_idx)
        #time
        self.writer.add_scalar('time/train', time_spent, self.niter, obj_idx)
        #regloss:
        self.writer.add_scalar('reg/train', loss_reg, self.niter, obj_idx)
        #loss_l2:
        self.writer.add_scalar('loss_l2/train', loss_l2, self.niter, obj_idx)
        #loss:
        self.writer.add_scalar('loss/train', loss, self.niter, obj_idx)

    def image_log(self, generated_img, gtimg, obj_idx):
        #Image: left:generated image, right:GT image
        H,W = generated_img.shape[:-1]
        compare_win = torch.zeros(H, 2*W, 3)
        compare_win[:,:W,:] = generated_img
        compare_win[:,W:,:] = gtimg
        compare_win = image_float_to_uint8(compare_win.detach().cpu().numpy())
        self.writer.add_image('train_'+str(self.niter) + '_' + str(obj_idx.item()),
                              torch.from_numpy(compare_win).permute(2,0,1))
