import torch.nn as nn
from stylegan2.model import ConvLayer, ResBlock, EqualLinear
import math
import torch.nn.functional as F
import numpy as np

from torchvision.utils import save_image
from criteria import id_loss


def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y


class GradualStyleEncoder2(nn.Module):
    def __init__(self, in_channel, latent, n_styles, size):
        super().__init__()
        # self.fc = self.
        self.in_channel = in_channel
        self.n_styles = n_styles
        self.latent = latent

        convs = []
        blur_kernel = [1, 3, 3, 1]

        num_pools = int(math.log(size, 2))
        for i in range(2):
            # convs.append(ResBlock(in_channel, in_channel, blur_kernel))
            convs.append(ConvLayer(in_channel, in_channel, 3, downsample=True))

        self.convs = nn.Sequential(*convs)

        self.fc = EqualLinear(self.in_channel * 4, latent * n_styles, lr_mul=0.01, activation="fused_lrelu")
        self.fsize = 8

    def forward(self, x):
        n = x.shape[0]

        x = F.interpolate(x, (self.fsize, self.fsize))
        x = self.convs(x)

        x = x.view(n, -1)
        x = self.fc(x)

        x = x.view(n, self.n_styles, self.latent)
        return x


class StyleEncoder(nn.Module):
    def __init__(self, size, style_dim=512, input_channel=3, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        channels = {k: v // 2 for k, v in channels.items()}

        convs = [ConvLayer(input_channel, channels[size], 1)]
        log_size = int(math.log(size, 2))
        self.style_count = 2 * log_size - 2

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.features_encoder = nn.Sequential(*convs)
        self.styles = nn.ModuleList()

        # self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

        self.latlayer1 = ConvLayer(256, 256, 3, downsample=False)
        self.latlayer2 = ConvLayer(256, 256, 3, downsample=False)

        self.coarse_ind = 3
        self.middle_ind = 7

        self.style1 = GradualStyleEncoder2(256, style_dim, self.coarse_ind, 16)
        self.style2 = GradualStyleEncoder2(256, style_dim, self.middle_ind - self.coarse_ind, 32)
        self.style3 = GradualStyleEncoder2(256, style_dim, self.style_count - self.middle_ind, 64)

        # for i in range(self.style_count):
        #     if i < self.coarse_ind:
        #         feature_size = 16
        #     elif i < self.middle_ind:
        #         feature_size = 32
        #     else:
        #         feature_size = 64
        #
        #     # num_channels = channels[feature_size]
        #     style = GradualStyleEncoder(256, 256, feature_size)
        #     self.styles.append(style)

    def forward(self, x):

        # from encoder4editing
        modulelist = list(self.features_encoder._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c1 = x
            elif i == 3:
                c2 = x
            elif i == 4:
                c3 = x

        # Infer main W and duplicate it
        w1 = self.style1(c3)

        p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
        w2 = self.style2(p2)

        p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
        w3 = self.style3(p1)

        out = torch.cat([w1, w2, w3], dim=1)
        return out


import pytorch_lightning as pl
from cgan import TeethGenerator
import torch
from encoders.psp_encoders import GradualStyleEncoder

class PSP(pl.LightningModule):
    def __init__(self, size=256, style_dim=256, n_mlp=8, num_labels=2, with_bg=True,ckpt=None, start_from_latent_avg=True):
        super(PSP, self).__init__()
        # self.psp_encoder = StyleEncoder(size, style_dim, )
        self.psp_encoder = GradualStyleEncoder(50, 'ir_se')
        self.decoder = TeethGenerator(size, style_dim, n_mlp=n_mlp, num_labels=num_labels, with_bg=with_bg)
        self.save_img = 0
        if ckpt is not None:
            ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
            self.decoder.load_state_dict(ckpt["g_ema"])

        self.start_from_latent_avg = start_from_latent_avg
        latent_avg = self.decoder.mean_latent(int(1e5))[0].detach()
        self.register_buffer('latent_avg', latent_avg)
        print('latent avg finish')

        self.start_from_latent_avg = True

        self.mse_loss = nn.MSELoss().to(self.device).eval()
        self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        # self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.start_from_latent_avg)
        self.id_loss = id_loss.IDLoss().to(self.device).eval()
        # self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()

        self.id_lambda = 0.2
        self.lpips_lambda = 0.8
        self.w_norm_lambda = 0.
        # self.w_norm_lambda = 0.0
        # self.l2_lambda = 100.
        self.l2_lambda = 1
        self.moco_lambda = 0.

    def forward(self, x, geometry, background, mask,
                input_code=False, randomize_noise=False,
                return_latents=False):
        codes = self.psp_encoder(x)
        # if self.start_from_latent_avg:
        #     if codes.ndim == 2:
        #         codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        #     else:
        #         codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             geometry=geometry,
                                             background=background,
                                             mask=mask,
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        return images, result_latent

    def training_step(self, batch, batch_idx):
        loss, _ = self.get_loss(batch, 'train')
        self.log('loss', loss)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     # 将验证集第一个batch的结果画在tensorboard里
    #     self.sample_dir = os.path.join(self.logger.log_dir, 'samples')
    #     os.makedirs(self.sample_dir, exist_ok=True)
    #     loss, (y, y_hat) = self.get_loss(batch, 'val')
    #     if batch_idx % 20 == 0:
    #         img = torch.cat([y, y_hat], dim=2)
    #         save_image(
    #             img,
    #             f"{self.sample_dir}/{str(self.global_step).zfill(5)}_{str(batch_idx).zfill(5)}.png",
    #             normalize=True,
    #             value_range=(-1, 1),
    #         )
    #     return loss

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
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

        # n, c, w, h = y.shape
        # y_flatten = y.view(n, c, -1)
        # yhat_flatten = y_hat.view(n, c, -1)
        #
        # mean_y = torch.mean(y_flatten, dim=-1)
        # mean_yhat = torch.mean(yhat_flatten, dim=-1)
        # mean_loss = F.l1_loss(mean_y, mean_yhat)
        #
        # std_y = torch.std(y_flatten, dim=-1)
        # std_yhat = torch.std(yhat_flatten, dim=-1)
        # std_loss = F.l1_loss(std_y, std_yhat)
        #
        # loss_dict['mean'] = 10*mean_loss
        # loss_dict['std'] = 10*std_loss
        # loss += mean_loss
        # loss += std_loss

        loss_dict['loss'] = float(loss)

        return loss, loss_dict, id_logs

    def get_loss(self, batch, mode):
        img = batch['img']
        mask = batch['mask']
        input_semantic = batch['input_semantic']
        background = batch['bg']

        # x = img * mask
        x = img
        y_hat, latent = self.forward(x, input_semantic, background, mask, return_latents=True)
        y = img
        # loss, loss_dict, id_logs = self.calc_loss(x, y * mask, y_hat * mask, latent)
        loss, loss_dict, id_logs = self.calc_loss(x, y , y_hat , latent)
        if mode == 'train':
            lr = [x['lr'] for x in self.optimizers().param_groups]  # for tensorboard
            self.log('lr', lr[0])
        else:
            self.log('val_loss', loss)
        return loss, (y, y_hat)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        if batch_idx % 50 == 0:
            data = np.load('./test.npz')
            mouth = torch.FloatTensor(data['mouth'].transpose(2, 0, 1)).unsqueeze(0).cuda()
            bg = torch.FloatTensor(data['bg'].transpose(2, 0, 1)).unsqueeze(0).cuda()
            mask = torch.FloatTensor(data['mask'].astype(np.float32)[None]).unsqueeze(0).cuda()
            input_semantic = torch.FloatTensor(data['input_semantic']).unsqueeze(0).cuda()
            image, _ = self.forward(mouth, input_semantic, bg, mask)
            save_image(
                image,
                os.path.join('media', f'{self.save_img}.png'),
                normalize=True,
                value_range=(-1, 1),
            )
            self.save_img += 1
    #
    # def on_epoch_end(self):
    #     torch.save(self.psp_encoder.state_dict(), os.path.join(self.logger.log_dir,'encoder.pkl'))

    def configure_optimizers(self):
        from torch.optim import Adam, lr_scheduler
        H = self.hparams
        opt = Adam(self.psp_encoder.parameters(), lr=0.0001)
        scheduler = lr_scheduler.OneCycleLR(opt, 0.0001, pct_start=0.1,
                                            total_steps=5000)
        return [opt], [{'scheduler': scheduler, 'interval': 'step'}]


from criteria import w_norm
from criteria.lpips.lpips import LPIPS
from stylegan2.dataset import less_conditional
from torch.utils.data import DataLoader
class SmileDM(pl.LightningDataModule):
    def __init__(self):
        super(SmileDM, self).__init__()

    def setup(self, stage=None):
        dataset = less_conditional()
        dataset_length = len(dataset)
        val_length = dataset_length // 20
        self.train_dataset = dataset
        self.num_workers = 6

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=2, shuffle=True,
                          num_workers=self.num_workers, pin_memory=False)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=4, shuffle=False,
    #                       num_workers=self.num_workers, pin_memory=False)


if __name__ == '__main__':
    import torch
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import ModelCheckpoint, Callback
    from pytorch_lightning import Trainer

    import os

    # name = 'encoder'
    # ckpt = r'weights\default\110000.pt'
    # num_labels = 8

    # name = 'smile_encoder'
    name = 'less_condition_up_encoder'
    ckpt = '/home/vsfh/training/less_cgan/down/020000.pt'
    num_labels = 2
    with_bg = True

    pl_model = PSP(256, 256, num_labels=num_labels, ckpt=ckpt, with_bg=with_bg)

    import time
    timestamp = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))

    accelerator = None

    checkpoint_callback = ModelCheckpoint(monitor='loss', save_last=False, save_top_k=20)
    logger = pl_loggers.TestTubeLogger('/home/vsfh/training/less_encoder/down', name=name+'_'+timestamp)
    trainer = pl.Trainer(gpus=1, max_epochs=2, callbacks=[checkpoint_callback], logger=logger)

    pl.seed_everything(666)
    dm = SmileDM()
    trainer.fit(pl_model, datamodule=dm)