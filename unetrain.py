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
from torch.utils.tensorboard import SummaryWriter
import torchvision

class MultiConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int):
        super(MultiConv, self).__init__()
        layers = [nn.Conv2d(in_ch, out_ch, (3, 3), (1, 1), 1), nn.ReLU(True)]
        for i in range(num_layers - 1):
            layers += [nn.Conv2d(out_ch, out_ch, (3, 3), (1, 1), 1), nn.ReLU(True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int):
        super(DownBlock, self).__init__()
        layers = [nn.MaxPool2d((2, 2), (2, 2)),
                  MultiConv(in_ch, out_ch, num_layers)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CenterBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(CenterBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(UpBlock, self).__init__()
        self.net = nn.Sequential(
            CenterBlock(in_ch, out_ch),
            CenterBlock(out_ch, out_ch)
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.net(x)


class SegmentationHead(nn.Module):
    def __init__(self, in_ch: int, classes: int, pretrained=True):
        super(SegmentationHead, self).__init__()
        self.UpBlock = UpBlock(128, 64)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, classes, (3, 3), (1, 1), 1),
            nn.Sigmoid()
        )
        if not pretrained:
            self.init_weight()

    def forward(self, x, info=None):
        x = self.UpBlock(x, None)
        if info:
            info.append(x)
            x = torch.cat(info, dim=1)
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, num_layers=3):
        super(UNet, self).__init__()
        self.num_layers = num_layers
        self.encoder = nn.ModuleList([MultiConv(3, 64, 2),
                                      DownBlock(64, 128, 2),
                                      DownBlock(128, 256, 2)])
        self.center = nn.Sequential(
            CenterBlock(256, 256))
        self.decoder = nn.ModuleList([UpBlock(384, 128),
                                    UpBlock(128, 2)])

    def forward(self, x):
        xi = [self.encoder[0](x)]
        for layer in self.encoder[1: self.num_layers + 1]:
            xi.append(layer(xi[-1]))
        xi[-1] = self.center(xi[-1])
        xi[0] = None
        for i, layer in enumerate(self.decoder):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return xi[-1]


    
def toy():
    # renderer = render_init()
    # # seg = seg_model()
    # seg = None
    save_path = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.23/edge'
    # file_path = '/mnt/share/shenfeihong/data/smile_/C01001659595'
    # upper, lower, mid = load_up_low(file_path, show=False)
    # mask = 1
    # angle = torch.tensor([-1.396,0,0]).unsqueeze(0).cuda()
    # movement = torch.tensor([0,0,475]).unsqueeze(0).cuda()
    # pred_tid_seg = render_(renderer=renderer, upper=upper, lower=lower, \
    #                             mask=mask, angle=angle, movement=movement)
    
    val_renderer = render_init('val')

    file_path = '/mnt/share/shenfeihong/data/smile_/C01001659595'
    movement = [0, 0, 470]
    # upper_ori, lower_ori, mid_ori = load_up_low(file_path, mode='ori', show=False)
    # edge_ori = render_(renderer,upper_ori, lower_ori, mid_ori,1, movement=movement)
    upper, lower, mid = load_up_low(file_path, mode='T2', num_teeth=4)
    upper = meshes_to_tensor(upper).cuda()
    lower = meshes_to_tensor(lower).cuda()
    angle = torch.tensor([-1.396,0,0]).unsqueeze(0).cuda()
    movement = torch.tensor([0,0,475]).unsqueeze(0).cuda()
    dist_down = torch.tensor([0,0,0]).unsqueeze(0).cuda()
    
    mask = 1
    deepmap_upper = render_(val_renderer, upper, lower, mid, mask, angle, movement, dist_down, mode='val')
    print(torch.unique(deepmap_upper), mid)
    deepmap_upper = deepmapToedge(deepmap_upper, mid)[None]
    
    cv2.imwrite(f'{save_path}/out_up.png', (deepmap_upper[0]).astype(np.uint8))
    
    print('done')

def model():
    model = UNet()
    print(model(torch.randn(3,3,256,256)).shape)
def train():
    
    save_path = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.23/edge'
    tensorboard_path = os.path.join(save_path, 'lightning_logs')
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(tensorboard_path)
    model = UNet().cuda()
    # ckpt = torch.load(os.path.join(save_path, '45.pt'))
    # model.load_state_dict(ckpt)

    optimize = optim.AdamW(model.parameters(), lr = 0.0001)

    dl = get_loader_unet(size=4, mode='train')
    val_dl = get_loader_unet(size=1, mode='val')
    # iteration = iter(dl)
    for index in range(50):
        if not index % 5:
            torch.save(model.state_dict(), f"{save_path}/{index}.pt")
        for idx,batch in enumerate(dl):

            optimize.zero_grad()
            mouth = batch['mouth'].cuda()
            up_edge = batch['up_edge'].cuda()
            mask = batch['mask'].cuda()
            label = batch['label'].type(torch.float32).cuda()
            
            zero_ = torch.tensor([0]).unsqueeze(0).repeat(mask.shape[0],1).cuda()

            output = model(mouth - up_edge)
            
            loss = F.mse_loss(output, label)-torch.sum(torch.abs(label[:,0,:,:]-label[:,1,:,:]))
            loss.backward()
            optimize.step()
            writer.add_scalar('Loss/train', loss.item(), idx)

            # writer.add_image('Image/Gap', mouth[0], idx)
            if not idx%100:
                writer.add_image('Image/label0', label[0][:1], idx)
                writer.add_image('Image/label1', label[0][1:], idx)
                
                writer.add_image('Image/output0', output[0][:1], idx)
                writer.add_image('Image/output1', output[0][1:], idx)
            
            
            
            if False:
                upper, lower, mid = load_up_low(file_path, mode='T2', num_teeth=4)
                upper = meshes_to_tensor(upper).cuda()
                lower = meshes_to_tensor(lower).cuda()
                for i, val_batch in enumerate(val_dl):
                    mouth = val_batch['mouth'].cuda()
                    mask = val_batch['mask'].cuda()
                    up_edge = val_batch['up_edge'].cuda()
                    down_edge = val_batch['down_edge'].cuda()
                    zero_ = torch.tensor([0]).unsqueeze(0).repeat(mask.shape[0],1).cuda()
                    with torch.no_grad():
                        # print(up_edge.shape, down_edge.shape, mask.shape)
                        output = model(torch.cat((up_edge, down_edge, mask),1))
                        dist_up = torch.cat((zero_,zero_,output[:,4:5]),1)
                        dist_down = torch.cat((zero_,zero_,output[:,3:4]),1)
                        angle = torch.cat((zero_-1.396,zero_,zero_),1)
                        movement = torch.cat((output[:,0:1],output[:,1:2],output[:,2:3]*10+470),1)
                        deepmap_upper = render_(val_renderer, upper, lower, mid, mask, angle, movement, [dist_up,dist_down], mode='val')

                    deepmap_edge = deepmapToedge(deepmap_upper, mid)[None]
                    edge_tensor = torch.tensor(deepmap_edge/255).cuda()

                    mouth = torch.clip((mouth[0]+1)/2,min=0,max=1)
                    mouth_edge = torch.cat((edge_tensor, mouth),0)
                    writer.add_image('Image/UpperOutput', deepmap_upper[None], i)
                    writer.add_image('Image/LowerOutput', mouth_edge, i)
                    writer.add_image('Image/Input', mouth, i)

if __name__=='__main__':
    # toy()
    # model()
    train()