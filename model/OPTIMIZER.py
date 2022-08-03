"""
Optimizing with GT pose
keeping the weights of the neural network fixed
optimize Zs_codes and Zt_codes
"""

import numpy as np
import torch
import json
from DATA import DATA_loader
from UTILS import get_rays, sample_from_rays, volume_rendering, image_float_to_uint8
from skimage.metrics import structural_similarity as compute_ssim
from MODEL import ConNeRF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import imageio
import time


class Optimizer(object):
    def __init__(self, saved_dir, gpu, instance_ids=[], splits='test',
                 jsonfile = 'config.json', batch_size=2048, num_optimizes = 200):
        """
        :param saved_dir: the directory of pre-trained model
        :param gpu: which GPU we would use
        :param instances_ids: the number of images for test-time optimization(ex : 000082.png)
        :param splits: test or val
        :param configfile: where the hyper-parameters are saved
        :param num_optimizes : number of test-time optimization steps
        """
        super().__init__()
        # Read Hyperparameters
        config_path = os.path.join('config_file', jsonfile)
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.device = torch.device('cuda:' + str(gpu))
        self.batch_size = batch_size
        self.num_optimizes = num_optimizes
        self.psnr_eval = {}
        self.psnr_optimize = {}
        self.ssim_eval = {}
        self.nviews = str(len(instance_ids))
        self.splits = splits

        self.make_dataloader(splits, len(instance_ids))
        self.make_model()
        self.load_model_codes(saved_dir)
        print('we are going to save at ', self.save_dir)
        #self.saved_dir = saved_dir



    def optimize_objs(self, instance_ids, lr=1e-2, lr_half_interval=50, save_img = True):
        #save optimize log file
        logpath = os.path.join(self.save_dir, 'optimize_config.json')
        hpam = {'instance_ids' : instance_ids, 'lr': lr, 'lr_half_interval': lr_half_interval,
                'self.splits': self.splits}
        with open(logpath, 'w') as f:
            json.dump(hpam, f, indent=2)

        self.lr, self.lr_half_interval, iters = lr, lr_half_interval, 0
        instance_ids = torch.tensor(instance_ids)

        #all instances' latent codes initiation
        self.optimized_Zs_codes = torch.zeros(len(self.dataloader), self.mean_shape.shape[1])
        self.optimized_Zt_codes = torch.zeros(len(self.dataloader), self.mean_texture.shape[1])

        # Per object
        for num_obj, obj in enumerate(self.dataloader):
            focal, H, W, imgs, poses, obj_idx = obj
            tgt_imgs, tgt_poses = imgs[0, instance_ids], poses[0, instance_ids]
            self.noptimizes, self.lr_half_interval = 0, lr_half_interval

            Zs_code = self.mean_shape.to(self.device).clone().detach().requires_grad_()
            Zt_code = self.mean_texture.to(self.device).clone().detach().requires_grad_()

            # First Optimize
            self.set_optimizers(Zs_code, Zt_code)
            while self.noptimizes < self.num_optimizes:
                self.optimizes.zero_grad()
                t1 = time.time()
                generated_imgs, gt_imgs = [], []

                for num, instance_id in enumerate(instance_ids):
                    tgt_img, tgt_pose = tgt_imgs[num].reshape(-1,3), tgt_poses[num]

                    rays_o, viewdir = get_rays(H.item(), W.item(), focal, tgt_pose)
                    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, self.config['near'], self.config['far'],
                                                            self.config['N_samples'])
                    loss_per_img, generated_img = [], []
                    for i in range(0, xyz.shape[0], self.batch_size):
                        sigmas, rgbs = self.model(xyz[i:i+self.batch_size].to(self.device),
                                                  viewdir[i:i+self.batch_size].to(self.device),
                                                  Zs_code,
                                                  Zt_code
                                                  )

                        rgb_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
                        #print(rgb_rays.shape, gt_img.shape)

                        # LOSS calculation
                        # Euclidean distance
                        loss_l2 = torch.mean((rgb_rays - tgt_img[i:i+self.batch_size].type_as(rgb_rays))**2)

                        if i == 0:
                            reg_loss = torch.norm(Zs_code, dim=-1) + torch.norm(Zt_code, dim=-1)
                            loss_reg = self.config['loss_reg_coef'] * torch.mean(reg_loss)
                            loss = loss_l2 + loss_reg
                        else:
                            loss = loss_l2

                        loss.backward()

                        #record
                        loss_per_img.append(loss_l2.item())
                        generated_img.append(rgb_rays) # save one image

                    #save all images
                    generated_imgs.append(torch.cat(generated_img).reshape(H,W,3))
                    gt_imgs.append(tgt_img.reshape(H,W,3))

                # update parameters
                self.optimizes.step()

                # log
                self.log_optimize_psnr_time(np.mean(loss_per_img), time.time() - t1, self.noptimizes + self.num_optimizes * num_obj,
                                       num_obj)
                self.log_regloss(reg_loss.item(), self.noptimizes, num_obj)


                if save_img:
                    self.save_img(generated_imgs, gt_imgs, self.ids[num_obj], self.noptimizes)

                self.noptimizes += 1
                print("noptimizes:",self.noptimizes)

                #refine learning rate
                if self.noptimizes % lr_half_interval == 0:
                    self.set_optimizers(Zs_code, Zt_code)
                    print("learning rate:", self.lr)

            # Then, Evaluate
            # use images which are not in instances
            with torch.no_grad():
                #print(gt_poses.shape)
                for num in range(250):
                    if num not in instance_ids:

                        tgt_img, tgt_pose = imgs[0,num].reshape(-1,3), poses[0, num]
                        rays_o, viewdir = get_rays(H.item(), W.item(), focal, poses[0, num])
                        xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, self.config['near'], self.config['far'],
                                                               self.config['N_samples'])

                        loss_per_img, generated_img = [], []

                        for i in range(0, xyz.shape[0], self.batch_size):
                            sigmas, rgbs = self.model(xyz[i:i+self.batch_size].to(self.device),
                                                      viewdir[i:i + self.batch_size].to(self.device),
                                                      Zs_code,
                                                      Zt_code
                                                      )

                            rgb_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))

                            # LOSS calculation
                            # Euclidean distance
                            loss_l2 = torch.mean((rgb_rays - tgt_img[i:i+self.batch_size].type_as(rgb_rays)) ** 2)
                            loss_per_img.append(loss_l2.item())

                            #save single image
                            generated_img.append(rgb_rays)

                        self.log_eval_psnr(np.mean(loss_per_img), num, num_obj)
                        self.log_compute_ssim(torch.cat(generated_img).reshape(H, W, 3), tgt_img.reshape(H, W, 3),
                                              num, num_obj)

                        if save_img:
                            self.save_img([torch.cat(generated_img).reshape(H,W,3)], [tgt_img.reshape(H,W,3)], self.ids[num_obj], num,
                                          optimize=False)

            # Save the optimized codes
            self.optimized_Zs_codes[num_obj] = Zs_code.detach().cpu()
            self.optimized_Zt_codes[num_obj] = Zt_code.detach().cpu()
            self.save_optimizes(num_obj)

    def save_optimizes(self, num_obj):
        saved_dict = {
            'ids': self.ids,
            'num_obj' : num_obj,
            'optimized_Zs_codes' : self.optimized_Zs_codes,
            'optimized_Zt_codes': self.optimized_Zt_codes,
            'psnr_eval': self.psnr_eval,
            'ssim_eval': self.ssim_eval
        }
        torch.save(saved_dict, os.path.join(self.save_dir, 'codes.pth'))
        print('We finished the optimization of ' + str(num_obj))

    def save_img(self, generated_imgs, gt_imgs, obj_id, instance_num, optimize=True):
        H, W = gt_imgs[0].shape[:2]
        nviews = int(self.nviews)
        if not optimize:
            nviews = 1

        generated_imgs = torch.cat(generated_imgs).reshape(nviews, H, W, 3)
        gt_imgs = torch.cat(gt_imgs).reshape(nviews, H, W, 3)

        compare_win = torch.zeros(nviews *H, 2 * W, 3)
        compare_win[:,:W,:] = generated_imgs.reshape(-1, W, 3)
        compare_win[:,W:,:] = gt_imgs.reshape(-1, W, 3)
        compare_win = image_float_to_uint8(compare_win.detach().cpu().numpy())

        save_img_dir = os.path.join(self.save_dir, obj_id)

        if not os.path.isdir(save_img_dir):
            os.makedirs(save_img_dir)
        if optimize:
            self.writer.add_image('optimize_' + self.nviews + '_' + str(instance_num),
                                  torch.from_numpy(compare_win).permute(2,0,1))
            imageio.imwrite(os.path.join(save_img_dir, 'optimize' + self.nviews + '_' + str(instance_num) + '.png'), compare_win)
        else:
            self.writer.add_image(self.nviews + '_' + str(instance_num),
                                  torch.from_numpy(compare_win).permute(2, 0, 1))
            imageio.imwrite(os.path.join(save_img_dir, str(instance_num) + '_' + self.nviews + '.png'), compare_win)

    def log_compute_ssim(self, generated_img, gt_img, niters, obj_idx):
        generated_img_np = generated_img.detach().cpu().numpy()
        gt_img_np = gt_img.detach().cpu().numpy()
        ssim = compute_ssim(generated_img_np, gt_img_np, multichannel=True)

        if niters == 0:
            self.ssim_eval[obj_idx] = [ssim]
        else:
            self.ssim_eval[obj_idx].append(ssim)

    def log_eval_psnr(self, loss_per_img, niters, obj_idx):
        psnr = -10 * np.log(loss_per_img) / np.log(10)
        if niters == 0:
            self.psnr_eval[obj_idx] = [psnr]
        else:
            self.psnr_eval[obj_idx].append(psnr)

    def log_optimize_psnr_time(self, loss_per_img, time_spent, niters, obj_idx):
        psnr = -10*np.log(loss_per_img) / np.log(10)
        self.writer.add_scalar('psnr_optimize/' + self.nviews + '/' + self.splits, psnr, niters, obj_idx)
        self.writer.add_scalar('time_optimize/' + self.nviews + '/' + self.splits, time_spent, niters, obj_idx)

    def log_regloss(self, loss_reg, niters, obj_idx):
        self.writer.add_scalar('reg/'  + self.nviews + '/' + self.splits, loss_reg, niters, obj_idx)

    def set_optimizers(self, Zs_code, Zt_code):
        lr = self.get_learning_rate()
        #print(lr)
        #keeping the weights of the neural network fixed
        self.optimizes = torch.optim.AdamW([
            {'params': Zs_code, 'lr': lr},
            {'params': Zt_code, 'lr': lr}
        ])

    def get_learning_rate(self):
        optimize_values = self.noptimizes // self.lr_half_interval
        lr = self.lr * 2**(-optimize_values)
        return lr

    def make_model(self):
        self.model = ConNeRF(**self.config['net_hyperparams']).to(self.device)

    def load_model_codes(self, saved_dir):
        saved_path = os.path.join('exps', saved_dir, 'models.pth')
        saved_data = torch.load(saved_path, map_location = torch.device('cpu'))
        self.make_save_img_dir(os.path.join('exps', saved_dir, 'test'))
        self.make_writer(saved_dir)
        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)

        #why mean?
        self.mean_shape = torch.mean(saved_data['Zs_code_params']['weight'], dim=0).reshape(1,-1)
        self.mean_texture = torch.mean(saved_data['Zt_code_params']['weight'], dim=0).reshape(1,-1)

    def make_writer(self, saved_dir):
        self.writer = SummaryWriter(os.path.join('exps', saved_dir, 'test', 'runs'))

    def make_save_img_dir(self, save_dir):
        save_dir_tmp = save_dir
        num = 2
        while os.path.isdir(save_dir_tmp):
            save_dir_tmp = save_dir + '_' + str(num)
            num += 1

        os.makedirs(save_dir_tmp)
        self.save_dir = save_dir_tmp
        #print(self.save_dir)

    def make_dataloader(self, splits, num_intances_per_obj, crop_img=False):
        cat = self.config['data']['cat']
        data_dir = self.config['data']['data_dir']
        obj = cat.split('_')[1]
        splits = obj + '_' + splits
        DATA = DATA_loader(data_dir = data_dir,
                  num_intances_per_obj = num_intances_per_obj,
                  splits=splits,
                  mode='test',
                  crop = crop_img)
        self.ids = DATA.ids
        self.dataloader = DataLoader(DATA, batch_size=1, num_workers =4, shuffle = False)

