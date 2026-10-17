from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.t2m_dataset import MotionDataset
from models.vae.model import VAE
from models.vae.trainer import VAETrainer
from options.vae_option import arg_parse
from utils.fixseed import fixseed
from utils.get_opt import get_opt


def load_vae(vae_opt, filename):
    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', filename), map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    print(f'Loading VAE Model {filename}')
    return model


if __name__ == "__main__":
    opt = arg_parse(is_train=False)
    fixseed(opt.seed)

    # options the model was trained with
    vae_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, "opt.txt"), opt.device)
    vae = load_vae(vae_opt, "net_best_val.tar").to(opt.device)

    # test data, normalized with the statistics saved at training time
    mean = np.load(pjoin(vae_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(vae_opt.meta_dir, 'std.npy'))
    test_dataset = MotionDataset(vae_opt, mean, std, pjoin(vae_opt.data_root, 'test.txt'))
    test_loader = DataLoader(test_dataset, batch_size=vae_opt.batch_size, drop_last=False, num_workers=opt.num_workers, shuffle=False)

    trainer = VAETrainer(vae_opt, vae)
    trainer.test(test_loader, test_dataset.inv_transform, save_dir=pjoin(vae_opt.save_root, 'eval'))
