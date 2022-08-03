import numpy as np
import torch
import json
import os,sys
import imageio
import argparse
# instance 0 and 3

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

        self.azimuth = 0.1
        self.elevation = 0.3
        self.distance = 1.2

        self.model = ConNeRF(**self.config['net_hyperparams']).to(self.device)
        self.genarate_codes()
        self.poses = self.poses_loader(self.azimuth,self.elevation,self.distance)
        self.intrinsic_loader()
        self.load_trained_model(self.saved_dir, self.latent_code_path)
        self.load_latent_code(self.latent_code_path)



    def poses_loader(self, azimuth, elevation, distance):
        #load all angle from dataset
        # poses_dir = "/home/zhuzhengming/NeRF-GAN/MY_NeRF/dataset/srn_cars/cars_train/173669aeeba92aaec4929b8bd41b8bc6/pose"
        #
        # txtfiles = np.sort([os.path.join(poses_dir, f.name) for f in os.scandir(poses_dir)])
        # posefiles = np.array(txtfiles)
        # srn_coords_trans = np.diag(np.array([1, -1, -1, 1]))
        # poses = []
        # for posefile in posefiles:
        #     pose = np.loadtxt(posefile).reshape(4, 4)
        #     poses.append(pose @ srn_coords_trans)
        # return torch.from_numpy(np.array(poses)).float()

        poses = []
        T = self.C2W(azimuth,elevation,distance)
        poses.append(T)

        return torch.from_numpy(np.array(poses)).float()



    def C2W(self, azimuth, elevation, distance):
        R = np.array([[-np.sin(azimuth), np.cos(azimuth), 0],
                      [-np.sin(elevation) * np.cos(azimuth), -np.sin(elevation) * np.sin(azimuth), np.cos(elevation)],
                      [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]])

        p = np.array([distance * np.cos(elevation) * np.cos(azimuth),
                      distance * np.cos(elevation) * np.sin(azimuth),
                      distance * np.sin(elevation)])

        T = np.c_[R.transpose(), p.transpose()]
        T = np.r_[T, np.array([[0,0,0,1]])]

        return T

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

    def load_trained_model(self, saved_dir, latent_code_path):

        if os.path.exists(os.path.join(saved_dir,"models.pth")):
            checkpoint_path = os.path.join(saved_dir,"models.pth")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_params'])
            self.model = self.model.to(self.device)

        else:
            print("No models.pth!!")


    def Zs_interpolation(self):
        num = 100
        NZs_codes = torch.randn(num, 256)
        for i in range(num):
            NZs_codes[i] = self.Zs_codes[0] * (i/num) + (1 - i/num)*self.Zs_codes[3]

        return NZs_codes

    def Zt_interpolation(self):
        num = 100
        NZt_codes = torch.randn(num, 256)
        for i in range(num):
            NZt_codes[i] = self.Zt_codes[0] * (i / num) + (1 - i / num) * self.Zt_codes[3]

        return NZt_codes


    def load_latent_code(self, latent_code_path):
        if os.path.exists(os.path.join(latent_code_path, "codes.pth")):
            optimize_latent_code = torch.load(os.path.join(latent_code_path, "codes.pth"))

            self.Zs_codes = optimize_latent_code["optimized_Zs_codes"].to(self.device)
            self.Zt_codes = optimize_latent_code["optimized_Zt_codes"].to(self.device)

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

    def save_img(self, generated_img, Z_id, mode):

        if mode == 'Zs_codes':
            save_img_dir = os.path.join(self.output_path, "Zs_codes")
            generated_img = image_float_to_uint8(generated_img.detach().cpu().numpy())
            imageio.imwrite(os.path.join(save_img_dir,str(Z_id) + '.png'),generated_img)
            print("save Zs_image" + str(Z_id) + "successfully")

        if mode == 'Zt_codes':
            save_img_dir = os.path.join(self.output_path, "Zt_codes")
            generated_img = image_float_to_uint8(generated_img.detach().cpu().numpy())
            imageio.imwrite(os.path.join(save_img_dir, str(Z_id) + '.png'), generated_img)
            print("save Zt_image" + str(Z_id) + "successfully")


    def interpolation_rendering(self):
        NZs_codes = self.Zs_interpolation().to(self.device)
        NZt_codes = self.Zt_interpolation().to(self.device)

        #don't calculate grad
        # Zs_codes interpolation
        with torch.no_grad():
            for Zs_id in range(NZs_codes.shape[0]):

                t_pose = self.poses
                rays_o, viewdir = get_rays(self.H, self.W, self.focal, t_pose)
                xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir,
                                                        self.config['near'], self.config['far'],self.config['N_samples'])

                generated_img = []
                for i in range(0, xyz.shape[0], self.batch_size):
                    sigmas, rgbs = self.model(xyz[i:i + self.batch_size].to(self.device),
                                              viewdir[i:i + self.batch_size].to(self.device),
                                              NZs_codes[Zs_id],
                                              NZt_codes[0]
                                              )

                    rbg_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
                    generated_img.append(rbg_rays)

                generated_img = torch.cat(generated_img).reshape(self.H,self.W,3)
                self.save_img(generated_img,Zs_id,mode='Zs_codes')

        #Zt_codes interpolation
        with torch.no_grad():
            for Zt_id in range(NZt_codes.shape[0]):

                t_pose = self.poses
                rays_o, viewdir = get_rays(self.H, self.W, self.focal, t_pose)
                xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir,
                                                        self.config['near'], self.config['far'],self.config['N_samples'])

                generated_img = []
                for i in range(0, xyz.shape[0], self.batch_size):
                    sigmas, rgbs = self.model(xyz[i:i + self.batch_size].to(self.device),
                                              viewdir[i:i + self.batch_size].to(self.device),
                                              NZs_codes[0],
                                              NZt_codes[Zt_id]
                                              )

                    rbg_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
                    generated_img.append(rbg_rays)

                generated_img = torch.cat(generated_img).reshape(self.H,self.W,3)
                self.save_img(generated_img,Zt_id,mode='Zt_codes')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="rendering_test")
    arg_parser.add_argument("--gpu", dest="gpu", default="0")
    arg_parser.add_argument("--saved_dir", dest="saved_dir", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/save_dir")
    arg_parser.add_argument("--output", dest="output", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/output_latent")
    arg_parser.add_argument("--jsonfile", dest="jsonfile", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/config_file/config.json")
    arg_parser.add_argument("--batch_size", dest="batch_size", default=2048)
    arg_parser.add_argument("--latent_code_path", dest="latent_code_path", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/save_dir/test")

    args = arg_parser.parse_args()
    saved_dir = args.saved_dir
    gpu = int(args.gpu)
    batch_size = int(args.batch_size)

    pose_render = pose_renderer(saved_dir, gpu, args.latent_code_path, args.jsonfile, batch_size, args.output)
    pose_render.interpolation_rendering()