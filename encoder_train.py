import torch.nn as nn
from stylegan2.model import ConvLayer, ResBlock, EqualLinear
import math
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms, utils
from criteria import id_loss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

import pytorch_lightning as pl
from cgan import TeethGenerator
from encoders.model import network as encoder 
import torch

class PSP(pl.LightningModule):
    def __init__(self, n_mlp=8, ckpt=None, start_from_latent_avg=True):
        super(PSP, self).__init__()
        self.psp_encoder = encoder(50, 'ir_se')
        self.decoder = TeethGenerator(256, 256, n_mlp=8)
        self.epoch_index = 0

        if ckpt is not None:
            ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
            self.decoder.load_state_dict(ckpt["g_ema"])

        self.mse_loss = nn.MSELoss().to(self.device).eval()
        self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        # self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.start_from_latent_avg)
        self.id_loss = id_loss.IDLoss().to(self.device).eval()
        # self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()

        self.id_lambda = 0.
        self.lpips_lambda = 0.8
        self.w_norm_lambda = 0.
        self.w_norm_lambda = 0.0
        self.l2_lambda = 1.0
        self.moco_lambda = 0.

    def forward(self, real_img, mask, big_mask=None, mix=False,
                input_code=False, randomize_noise=False,
                return_latents=False):
        
        codes = self.psp_encoder(real_img)
        input_is_latent = not input_code
        images, result_latent = self.decoder([codes], real_image=real_img, mask=mask,
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)
        # images = real_img*(1-big_mask)+images*big_mask
        return images, result_latent

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        return loss

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.id_lambda > 0:
            loss_id,_,_ = self.id_loss(y_hat, y)
            loss_dict['loss_id'] = float(loss_id)
            loss = loss_id * self.id_lambda
        if self.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.l2_lambda
        if self.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.lpips_lambda
        if self.w_norm_lambda > 0:
            loss_w_norm = self.w_norm_loss(latent, self.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.w_norm_lambda
        if self.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.moco_lambda

        loss_dict['loss'] = float(loss)

        return loss, loss_dict, id_logs

    def get_loss(self, batch, mode):
        img = batch['images']
        mask = batch['mask']
        # big_mask = batch['big_mask']
        

        y_hat, latent = self.forward(img, mask, return_latents=True)

        loss, loss_dict, id_logs = self.calc_loss(img, img, y_hat , latent)
        if not mode%100:
            utils.save_image(
                y_hat,
                f"./2022.12.13/encoder/{mode}.png",
                nrow=2,
                normalize=True,
                range=(-1, 1),
            )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        pass

    def on_epoch_end(self):
        self.epoch_index += 1
        torch.save(self.psp_encoder.state_dict(), os.path.join('./2022.12.13/encoder/ckpt',f'{self.epoch_index}.pt'))

    def configure_optimizers(self):
        from torch.optim import AdamW
        self.requires_grad(self.decoder, False)
        opt = AdamW(self.psp_encoder.parameters(), lr=0.0001)
        return opt
    
    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

from criteria import w_norm
from criteria.lpips.lpips import LPIPS
from stylegan2.dataset import *
from torch.utils.data import DataLoader
class SmileDM(pl.LightningDataModule):
    def __init__(self):
        super(SmileDM, self).__init__()

    def setup(self, stage=None):
        dataset = fuck('encoder')
        self.train_dataset = dataset
        self.num_workers = 8
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=4, shuffle=True,
                          num_workers=self.num_workers, pin_memory=False)

if __name__ == '__main__':
    import torch
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import ModelCheckpoint, Callback
    from pytorch_lightning import Trainer

    name = 'no_condition_encoder'
    ckpt = '/mnt/share/shenfeihong/tmp/wo_edge/040000.pt'

    pl_model = PSP(ckpt=ckpt)
    accelerator = None
    gpus = 1
    epoch = 20
    trainer = Trainer(
        gpus=gpus,
        max_steps=20000,
        precision=16,
        flush_logs_every_n_steps=10,
        num_sanity_val_steps=0,
        val_check_interval=199,
        log_every_n_steps=1,
        profiler='simple',
    )
    pl.seed_everything(666)
    dm = SmileDM()
    trainer.fit(pl_model, datamodule=dm)
