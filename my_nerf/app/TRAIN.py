import argparse
import sys

sys.path.append('/home/zhuzhengming/NeRF-GAN/MY_NeRF/model')
from TRAINER import Trainer

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='My_conNeRF')
    arg_parser.add_argument("--gpu", dest="gpu", default="0")
    arg_parser.add_argument("--save_dir", dest="save_dir", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/save_dir")
    arg_parser.add_argument("--iters", dest="iters", default=200000)
    arg_parser.add_argument("--batch_size", dest="batch_size", default=1024)
    arg_parser.add_argument("--jsonfile_dir", dest="jsonfile_dir", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/config_file/config.json")
    arg_parser.add_argument("--num_instances_per_obj", dest="num_instances_per_obj", default=2)
    arg_parser.add_argument("--mode", dest="mode", default="train")
    arg_parser.add_argument("--check_iter", dest="check_iter", default=2000)
    arg_parser.add_argument("--checkpoints_path", dest="checkpoints_path", default="/home/zhuzhengming/NeRF-GAN/MY_NeRF/save_dir")

    args = arg_parser.parse_args()
    save_dir = args.save_dir
    gpu = int(args.gpu)
    iters = int(args.iters)
    batch_size = int(args.batch_size)
    jsonfile = args.jsonfile_dir
    num_instances_per_obj = int(args.num_instances_per_obj)
    check_iter = args.check_iter
    checkpoints_path = args.checkpoints_path

    trainer = Trainer(save_dir, gpu, checkpoints_path, jsonfile, batch_size, check_iter)
    trainer.training(iters, num_instances_per_obj)
