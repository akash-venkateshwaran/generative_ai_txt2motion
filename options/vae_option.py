import argparse
import os
import torch
from os.path import join as pjoin
from utils import paramUtil

def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## basic setup
    parser.add_argument("--name", type=str, default="vae_default", help="Name of this trial")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id (-1 for cpu)")

    ## dataloader
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--window_size", type=int, default=64, help="training motion length")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader")

    ## optimization
    parser.add_argument("--max_epoch", default=50, type=int, help="number of total epochs to run")
    parser.add_argument("--warm_up_iter", default=2000, type=int, help="number of total iterations for warmup")
    parser.add_argument("--lr", default=2e-4, type=float, help="max learning rate")
    parser.add_argument("--milestones", default=None, nargs="+", type=int, help="lr decay iterations (default: 60%% and 85%% of total iterations)")
    parser.add_argument("--gamma", default=0.05, type=float, help="learning rate decay")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")

    parser.add_argument("--patience", default=0, type=int, help="stop after n epochs without val improvement (0 = off)")

    parser.add_argument("--recon_loss", type=str, default="l1_smooth", choices=["l1", "l1_smooth"], help="reconstruction loss")
    parser.add_argument("--lambda_pos", type=float, default=0.5, help="position loss weight")
    parser.add_argument("--lambda_vel", type=float, default=0.5, help="velocity loss weight")
    parser.add_argument("--lambda_kl", type=float, default=0.02, help="kl loss weight")

    ## vae arch
    parser.add_argument("--latent_dim", type=int, default=32, help="embedding dimension")
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size")
    parser.add_argument("--n_layers", type=int, default=2, help="num of layers")
    parser.add_argument("--n_extra_layers", type=int, default=1, help="num of extra layers")
    parser.add_argument("--norm", type=str, default="none", help="normalization", choices=["none", "batch", "layer"])
    parser.add_argument("--activation", type=str, default="gelu", help="activation function", choices=["relu", "silu", "gelu"])
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")

    ## other
    parser.add_argument("--is_continue", action="store_true", help="Resume from latest.tar of this trial")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here")
    parser.add_argument("--log_every", default=10, type=int, help="iter log frequency")
    parser.add_argument("--save_latest", default=500, type=int, help="iter save latest model frequency")
    parser.add_argument("--tb_port", default=6006, type=int, help="port for the TensorBoard server started by train_vae.py (0 = don't start)")

    opt = parser.parse_args()
    opt.dataset_name = "kit"  # only KIT-ML is supported; kept as an attribute for checkpoint/log paths
    if opt.gpu_id >= 0:
        torch.cuda.set_device(opt.gpu_id)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    for d in (opt.model_dir, opt.meta_dir, opt.log_dir):
        os.makedirs(d, exist_ok=True)

    args = vars(opt).copy()  # options saved to opt.txt; the dataset constants below are not saved

    # KIT-ML constants (not saved)
    opt.data_root = './dataset/KIT-ML/'
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.joints_num = 21
    opt.pose_dim = 251
    opt.contact_joints = [19, 20, 14, 15]
    opt.fps = 12.5
    opt.radius = 240 * 8
    opt.kinematic_chain = paramUtil.kit_kinematic_chain

    opt_str = '------------ Options -------------\n'
    opt_str += ''.join('%s: %s\n' % (k, v) for k, v in sorted(args.items()))
    opt_str += '-------------- End ----------------\n'
    print(opt_str, end='')

    opt.is_train = is_train
    if is_train:
        with open(pjoin(opt.save_root, 'opt.txt'), 'wt') as opt_file:
            opt_file.write(opt_str)

    return opt
