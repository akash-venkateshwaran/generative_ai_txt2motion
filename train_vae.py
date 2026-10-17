import os
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import DataLoader

from options.vae_option import arg_parse
from models.vae.model import VAE
from models.vae.trainer import VAETrainer
from data.t2m_dataset import MotionDataset

from utils.fixseed import fixseed

os.environ["OMP_NUM_THREADS"] = "1"


def start_tensorboard(log_dir, port):
    from tensorboard import program
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, '--port', str(port)])
    print('TensorBoard: %s' % tb.launch())


if __name__ == "__main__":
    opt = arg_parse(True)
    fixseed(opt.seed)

    if opt.tb_port > 0:
        start_tensorboard(opt.log_dir, opt.tb_port)

    # model
    net = VAE(opt)
    num_params = sum(param.numel() for param in net.parameters())
    print('Total trainable parameters: {}M'.format(num_params / 1_000_000))

    # normalization statistics (see data/compute_kit_stats.py); a copy is kept with the checkpoints
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))
    np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
    np.save(pjoin(opt.meta_dir, 'std.npy'), std)

    # dataset & dataloader
    train_dataset = MotionDataset(opt, mean, std, pjoin(opt.data_root, 'train.txt'))
    val_dataset = MotionDataset(opt, mean, std, pjoin(opt.data_root, 'val.txt'))

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers, shuffle=False, pin_memory=True)

    # train
    trainer = VAETrainer(opt, net)
    trainer.train(train_loader, val_loader)

## python train_vae.py --name vae_kit_v1
