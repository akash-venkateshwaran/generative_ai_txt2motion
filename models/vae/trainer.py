import torch
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

import os
import time
from collections import OrderedDict, defaultdict

from utils.metrics import calculate_mpjpe
from utils.motion_process import recover_from_ric
from utils.utils import print_current_loss

def def_value():
    return 0.0

class VAETrainer:
    def __init__(self, opt, vae):
        self.opt = opt
        self.vae = vae

        if opt.is_train:
            self.logger = SummaryWriter(opt.log_dir)
        self.recon_criterion = {"l1": torch.nn.L1Loss, "l1_smooth": torch.nn.SmoothL1Loss}[opt.recon_loss]()

    def train_forward(self, batch_data):
        motion = batch_data.to(self.opt.device, dtype=torch.float32)
        J = self.opt.joints_num
        split = [4, 3 * (J - 1), 6 * (J - 1), 3 * J, 4]
        root, ric, rot, vel, contact = torch.split(motion, split, dim=-1)

        pred_motion, loss_dict = self.vae.forward(motion)
        pred_root, pred_ric, pred_rot, pred_vel, pred_contact = torch.split(pred_motion, split, dim=-1)

        self.motion = motion
        self.pred_motion = pred_motion

        # loss
        loss_rec = self.recon_criterion(pred_motion, motion)
        loss_vel = self.recon_criterion(pred_vel, vel)
        loss_pos = self.recon_criterion(pred_ric, ric)
        loss_kl = loss_dict["loss_kl"]

        loss = loss_rec + loss_vel * self.opt.lambda_vel + loss_pos * self.opt.lambda_pos + loss_kl * self.opt.lambda_kl

        loss_dict["loss_recon"] = loss_rec
        loss_dict["loss_vel"] = loss_vel
        loss_dict["loss_pos"] = loss_pos

        return loss, loss_dict

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optim.param_groups:
            param_group["lr"] = current_lr

    def save(self, file_name, epoch, total_iter):
        state = {
            "vae": self.vae.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "total_iter": total_iter,
            "best_val": self.best_val,
            "bad_epochs": self.bad_epochs,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.opt.device)
        self.vae.load_state_dict(checkpoint["vae"])
        self.optim.load_state_dict(checkpoint["optim"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_val = checkpoint["best_val"]
        self.bad_epochs = checkpoint["bad_epochs"]
        return checkpoint["epoch"], checkpoint["total_iter"]

    def train(self, train_loader, val_loader):
        self.vae.to(self.opt.device)

        total_iters = self.opt.max_epoch * len(train_loader)
        milestones = self.opt.milestones or [int(0.6 * total_iters), int(0.85 * total_iters)]
        milestones = [max(m - self.opt.warm_up_iter, 1) for m in milestones]  # the scheduler only steps after warm-up

        # optimizer
        self.optim = torch.optim.AdamW(self.vae.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        self.best_val = float("inf")
        self.bad_epochs = 0
        if self.opt.is_continue:
            epoch, it = self.resume(pjoin(self.opt.model_dir, 'latest.tar'))
            print("Load model epoch:%d iterations:%d" % (epoch, it))

        start_time = time.time()
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}, LR milestones (after warm-up): {milestones}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        # training loop
        while epoch < self.opt.max_epoch:
            self.vae.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                # forward
                self.optim.zero_grad()
                loss, loss_dict = self.train_forward(batch_data)
                loss.backward()
                self.optim.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()

                # log
                logs["loss"] += loss.item()
                logs["lr"] += self.optim.param_groups[0]['lr']
                for tag, value in loss_dict.items():
                    logs[tag] += value.item()

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1

            # validation
            print('Validation time:')
            self.vae.eval()
            val_log = defaultdict(def_value, OrderedDict())
            with torch.no_grad():
                for batch_data in val_loader:
                    loss, loss_dict = self.train_forward(batch_data)

                    val_log["loss"] += loss.item()
                    for tag, value in loss_dict.items():
                        val_log[tag] += value.item()

            msg = "Validation loss: "
            for tag, value in val_log.items():
                val_log[tag] = value / len(val_loader)
                self.logger.add_scalar('Val/%s' % tag, val_log[tag], epoch)
                msg += "%s: %.4f, " % (tag, val_log[tag])
            print(msg)

            # best checkpoint on validation loss
            if val_log["loss"] < self.best_val:
                print("--> Val loss improved from %.5f to %.5f" % (self.best_val, val_log["loss"]))
                self.best_val = val_log["loss"]
                self.bad_epochs = 0
                torch.save({'vae': self.vae.state_dict(), 'epoch': epoch}, pjoin(self.opt.model_dir, 'net_best_val.tar'))
            else:
                self.bad_epochs += 1

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if self.opt.patience > 0 and self.bad_epochs >= self.opt.patience:
                print("Early stopping: no val improvement for %d epochs" % self.bad_epochs)
                break

    @torch.no_grad()
    def test(self, test_loader, inv_transform, save_dir):
        """Reconstruction losses (normalized space) and MPJPE (de-normalized joint positions, root-aligned)."""
        os.makedirs(save_dir, exist_ok=True)
        self.vae.eval()

        log = defaultdict(def_value, OrderedDict())
        mpjpe, num_poses = 0.0, 0
        for batch_data in test_loader:
            loss, loss_dict = self.train_forward(batch_data)
            log["loss"] += loss.item()
            for tag, value in loss_dict.items():
                log[tag] += value.item()

            gt = recover_from_ric(torch.from_numpy(inv_transform(self.motion.cpu().numpy())).float(), self.opt.joints_num)
            pred = recover_from_ric(torch.from_numpy(inv_transform(self.pred_motion.cpu().numpy())).float(), self.opt.joints_num)
            gt, pred = gt.flatten(0, 1), pred.flatten(0, 1)  # [B * T, J, 3]
            mpjpe += torch.sum(calculate_mpjpe(gt, pred)).item()
            num_poses += gt.shape[0]

        msg = "".join("\t%s: %.4f\n" % (tag, value / len(test_loader)) for tag, value in log.items())
        msg += "\tMPJPE: %.4f\n" % (mpjpe / num_poses)
        print(msg)
        with open(pjoin(save_dir, 'eval.log'), 'w') as f:
            f.write(msg)
