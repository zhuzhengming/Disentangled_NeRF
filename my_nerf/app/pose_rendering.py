import numpy as np
import torch
import json
import os,sys
import imageio
import argparse


sys.path.append('/home/zhuzhengming/NeRF-GAN/MY_NeRF/model')

from UTILS import get_rays, sample_from_rays, volume_rendering, image_float_to_uint8
from MODEL import ConNeRF

class pose_renderer(object):
    def __init__(self,saved_dir, gpu,
                 latent_code_path,
                 jsonfile = 'config.json',
                 batch_size=2048,
                 output_path = "output"
                 ):
        super().__init__()

        # load Hyperparameters
        config_path = os.path.join('config_file', jsonfile)
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.saved_dir = saved_dir
        self.latent_code_path = latent_code_path
        self.device = torch.device('cuda:' + str(gpu))
        self.batch_size = batch_size
        self.output_path = output_path

        self.model = ConNeRF(**self.config['net_hyperparams']).to(self.device)
        self.genarate_codes()
        self.poses = self.poses_loader()
        self.intrinsic_loader()
        self.load_trained_model(self.saved_dir)
        self.load_latent_code(self.latent_code_path)



    def poses_loader(self):
        #load all angle from dataset
        poses_dir = "/home/zhuzhengming/NeRF-GAN/MY_NeRF/dataset/srn_cars/cars_train/173669aeeba92aaec4929b8bd41b8bc6/pose"

        txtfiles = np.sort([os.path.join(poses_dir, f.name) for f in os.scandir(poses_dir)])
        posefiles = np.array(txtfiles)
        srn_coords_trans = np.diag(np.array([1, -1, -1, 1]))
        poses = []
        for posefile in posefiles:
            pose = np.loadtxt(posefile).reshape(4, 4)
            poses.append(pose @ srn_coords_trans)
        return torch.from_numpy(np.array(poses)).float()

    def intrinsic_loader(self):
        # load all angle from dataset
        poses_dir = os.path.join("/home/zhuzhengming/NeRF-GAN/MY_NeRF/dataset/srn_cars/cars_train/173669aeeba92aaec4929b8bd41b8bc6",
                                 'intrinsics.txt')
        with open(poses_dir, 'r') as f:
            lines = f.readlines()
            focal = float(lines[0].split()[0])
            H, W = lines[-1].split()
            H, W = int(H), int(W)

            self.H = H
            self.W = W
            self.focal = focal

    def load_trained_model(self, saved_dir):

        if os.path.exists(os.path.join(saved_dir,"models.pth")):
            checkpoint_path = os.path.join(saved_dir,"models.pth")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_params'])
            self.model = self.model.to(self.device)

            self.Zs_codes = checkpoint['Zs_code_params']['weight']
            self.Zt_codes = checkpoint['Zt_code_params']['weight']

        else:
            print("No models.pth!!")

    def load_latent_code(self, latent_code_path):
        if os.path.exists(os.path.join(latent_code_path, "codes.pth")):
            optimize_latent_code = torch.load(os.path.join(latent_code_path, "codes.pth"))

            # self.Zs_codes = optimize_latent_code["optimized_Zs_codes"].to(self.device)
            # self.Zt_codes = optimize_latent_code["optimized_Zt_codes"].to(self.device)

        else:
            print("No codes.pth!!")


    def genarate_codes(self):
        embdim = self.config['net_hyperparams']['latent_dim']
        d = 1030

        self.Zs_codes = torch.randn(d, embdim)
        self.Zt_codes = torch.randn(d, embdim)
        # print(self.Zt_codes.shape)

        self.Zs_codes = self.Zs_codes.to(self.device)
        self.Zt_codes = self.Zt_codes.to(self.device)

    def save_img(self, generated_img, pose_id):
        save_img_dir = self.output_path
        generated_img = image_float_to_uint8(generated_img.detach().cpu().numpy())
        imageio.imwrite(os.path.join(save_img_dir,str(pose_id) + '.png'),generated_img)
        print("save image" + str(pose_id) + "successfully")

    def poses_rendering(self):
        #don't calculate grad
        with torch.no_grad():
            for pose_id in range(self.Zs_codes.shape[0]):
                # generated_imgs = []
                t_pose = self.poses[4]
                # H, W = t_pose.shape[:2]
                # print(self.H, self.W, self.focal)
                rays_o, viewdir = get_rays(self.H, self.W, self.focal, t_pose)
                xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir,
                                                        self.config['near'], self.config['far'],self.config['N_samples'])

                # print(self.Zs_codes[pose_id])
                generated_img = []
                for i in range(0, xyz.shape[0], self.batch_size):
                    sigmas, rgbs = self.model(xyz[i:i + self.batch_size].to(self.device),
                                              viewdir[i:i + self.batch_size].to(self.device),
                                              self.Zs_codes[pose_id],
                                              self.Zt_codes[5]
                                              )

                    rbg_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
                    generated_img.append(rbg_rays)

                generated_img = torch.cat(generated_img).reshape(self.H,self.W,3)
                self.save_img(generated_img,pose_id)




if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="rendering_test")
    arg_parser.add_argument("--gpu", dest="gpu", default="0")
    arg_parser.add_argument("--saved_dir", dest="saved_dir", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/save_dir")
    arg_parser.add_argument("--output", dest="output", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/output")
    arg_parser.add_argument("--jsonfile", dest="jsonfile", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/config_file/config.json")
    arg_parser.add_argument("--batch_size", dest="batch_size", default=2048)
    arg_parser.add_argument("--latent_code_path", dest="latent_code_path", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/save_dir/test")

    args = arg_parser.parse_args()
    saved_dir = args.saved_dir
    gpu = int(args.gpu)
    batch_size = int(args.batch_size)

    pose_render = pose_renderer(saved_dir, gpu, args.latent_code_path, args.jsonfile, batch_size, args.output)
    pose_render.poses_rendering()