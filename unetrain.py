from stylegan2.dataset import *
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from forward_render import *
import timm
from torch import optim

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # print(x4.shape)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Discriminator(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(Discriminator,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.final_conv = nn.Conv2d(512, 1, kernel_size=3, padding=1, bias=False)
        self.final_linear = nn.Linear(32, 1)
        self.final_linear1 = nn.Linear(32, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.final_conv(x4)[:,0,:,:]
        x6 = self.final_linear(x5)[...,0]
        x7 = self.final_linear1(x6)[...,0]
        return x7

class cls(nn.Module):
    def __init__(self, n_channels=2, n_classes=1, bilinear=False):
        super(cls,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.final_conv = nn.Conv2d(512, 1, kernel_size=3, padding=1, bias=False)
        self.final_linear = nn.Linear(32, 1)
        self.final_linear1 = nn.Linear(32, 6)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.final_conv(x4).squeeze()
        x6 = self.final_linear(x5).squeeze()
        x7 = self.final_linear1(x6)
        return x7
            
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag  
          
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss
         
def generate():
    from torch import optim
    save_path = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.15/edge'
    generator  = UNet().cuda()
    discriminator =Discriminator().cuda()
    g_optim = optim.AdamW(generator.parameters(), lr = 0.0001)
    d_optim = optim.AdamW(discriminator.parameters(), lr = 0.0001)
    dl = get_loader_unet(mode='train')
    val_dl = get_loader_unet(mode='val')
    iteration = iter(dl)
    for i in range(1000):
        for batch in dl:

            
            mask = batch['mask'].cuda()
            edge = batch['edge'].cuda()
            label = batch['label'].cuda()
            input_mask = torch.cat((mask, edge), 1)
            
            requires_grad(generator, False)
            requires_grad(discriminator, True)

            output = generator(input_mask)
            
            fake_pred = discriminator(output)
            real_pred = discriminator(label)
            d_loss = d_logistic_loss(real_pred, fake_pred)  
            
            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()
            
            requires_grad(generator, True)
            requires_grad(discriminator, False)

            output = generator(input_mask)

            fake_pred = discriminator(output)
            g_loss = g_nonsaturating_loss(fake_pred)*0.01
            edge_loss = F.mse_loss(output*mask, label*mask)*mask.mean()
            # print(g_loss, edge_loss)
            
            all_loss = edge_loss+g_loss
            
            generator.zero_grad()
            all_loss.backward()
            g_optim.step()

        if i >= 10 and not i % 5:
            j= 0
            for batch in val_dl:  
                mask = batch['mask'].cuda()
                edge = batch['edge'].cuda()
      
                input_mask = torch.cat((mask, edge), 1)
                with torch.no_grad():
                    output = generator(input_mask)
                utils.save_image(
                    output*mask,
                    f"{save_path}/sample/out_{i}_{j}.png",
                    nrow=1,
                    normalize=True,
                    range=(0, 1),
                )
                utils.save_image(
                    edge,
                    f"{save_path}/sample/input_{i}_{j}.png",
                    nrow=1,
                    normalize=True,
                    range=(0, 1),
                )                
                j+=1
            torch.save(generator.state_dict(), f"{save_path}/{i}.pt")

def toy():
    renderer = render_init()
    # seg = seg_model()
    seg = None
    save_path = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.23/edge'
    file_path = '/mnt/share/shenfeihong/data/smile_/C01001659595'
    upper, lower, mid = load_up_low(file_path, show=False)
    angle = torch.tensor([-1.396,0,0]).unsqueeze(0).repeat(6,1).cuda()
    movement = torch.tensor([0,0,475]).unsqueeze(0).repeat(6,1).cuda()
    dist = torch.tensor([0,0,0]).unsqueeze(0).repeat(6,1).cuda()
    pred_edge = render_(renderer=renderer, seg_model=seg, upper=upper, lower=lower, mask=1, angle=angle, movement=movement, dist=dist)
    # utils.save_image(
    #     pred_edge,
    #     f"{save_path}/sample/out_.png",
    #     nrow=1,
    #     normalize=True,
    #     range=(0, 1),
    # )

def train():

    save_path = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.23/edge'
    model = timm.create_model('efficientnetv2_l', in_chans=3, num_classes=6).cuda()
    renderer = render_init()
    seg = None
    file_path = '/mnt/share/shenfeihong/data/smile_/C01001659595'
    upper, lower, mid = load_up_low(file_path, show=False)
    optimize = optim.AdamW(model.parameters(), lr = 0.0001)

    dl = get_loader_unet(mode='train')
    # val_dl = get_loader_unet(mode='val')
    # iteration = iter(dl)
    for i in range(200):
        for batch in dl:

            optimize.zero_grad()
            mask = batch['mask'].cuda()
            up_edge = batch['up_edge'].cuda()
            
            down_edge = batch['down_edge'].cuda()

            # label = batch['label'].cuda()
            
            zero_ = torch.tensor([0]).unsqueeze(0).repeat(mask.shape[0],1).cuda()
            input_mask = torch.cat((mask, up_edge, down_edge), 1)

            output = model(input_mask)
            dist_up = torch.cat((zero_,zero_,zero_),1)
            dist_down = torch.cat((zero_,zero_,output[:,3:4]),1)
            angle = torch.cat((zero_-1.396,zero_,zero_),1)
            movement = torch.cat((output[:,0:1],output[:,1:2],output[:,2:3]*10+300),1)
            
            pred_up_edge, pred_down_edge = render_(renderer=renderer, seg_model=seg, upper=upper, lower=lower, mask=mask, angle=angle, movement=movement, dist=[dist_up, dist_down])
            # pred_down_edge_minus = torch.clip((pred_down_edge-pred_up_edge),min=0, max=1)
            # loss = F.mse_loss(torch.cat((pred_up_edge, pred_down_edge), 1), torch.cat((up_edge, down_edge), 1))
            loss = F.mse_loss(torch.clip((pred_up_edge+ pred_down_edge), min=0, max=1), torch.clip((up_edge+ down_edge), min=0, max=1))
            loss.backward()
            optimize.step()

        if not i % 2:
            print(movement[0][0],movement[1][0])
            utils.save_image(
                pred_up_edge[:4,...],
                f"{save_path}/up_pred/u_{i}.png",
                nrow=1,
                normalize=True,
                range=(0, 1),
            )
            utils.save_image(
                pred_down_edge[:4,...],
                f"{save_path}/down_pred/d_{i}.png",
                nrow=1,
                normalize=True,
                range=(0, 1),
            )
            utils.save_image(
                (down_edge)[:4,...],
                f"{save_path}/down/i_{i}.png",
                nrow=1,
                normalize=True,
                range=(0, 1),
            )                
            utils.save_image(
                (up_edge)[:4,...],
                f"{save_path}/up/i_{i}.png",
                nrow=1,
                normalize=True,
                range=(0, 1),
            )         
        if not i % 2:
            torch.save(model.state_dict(), f"{save_path}/{i}.pt")


if __name__=='__main__':
    # toy()
    
    train()