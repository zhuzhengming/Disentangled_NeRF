import sys, os
import argparse

sys.path.append('/home/zhuzhengming/NeRF-GAN/MY_NeRF/model')

from UTILS import str2bool
from OPTIMIZER import Optimizer



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="My_conNeRF")
    arg_parser.add_argument("--gpu",dest="gpu",default="0")
    arg_parser.add_argument("--saved_dir", dest="saved_dir", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/save_dir")
    arg_parser.add_argument("--gt_instances", dest = "gt_instances", nargs='+', default={1,17})
    arg_parser.add_argument("--splits", dest="splits", default='test')
    arg_parser.add_argument("--num_optimizes", dest="num_optimizes", default = 200)
    arg_parser.add_argument("--lr", dest="lr", default=1e-2)
    arg_parser.add_argument("--lr_half_interval", dest="lr_half_interval", default=50)
    arg_parser.add_argument("--save_img", dest="save_img", default=True)
    arg_parser.add_argument("--jsonfile", dest="jsonfile", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/config_file/config.json")
    arg_parser.add_argument("--batchsize", dest="batchsize", default=1024)

    args = arg_parser.parse_args()
    saved_dir = args.saved_dir
    gpu = int(args.gpu)
    lr = float(args.lr)
    lr_half_interval = int(args.lr_half_interval)
    save_img = str2bool(args.save_img)
    batchsize = int(args.batchsize)
    gt_instances = list(args.gt_instances)
    num_optimizes = int(args.num_optimizes)
    for num, i in enumerate(gt_instances):
        gt_instances[num] = int(i)




    optimizer = Optimizer(saved_dir, gpu, gt_instances, args.splits, args.jsonfile, batchsize, num_optimizes)
    optimizer.optimize_objs(gt_instances, lr, lr_half_interval, save_img)

