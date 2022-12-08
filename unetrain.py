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


def train():
    
    save_path = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.23/edge'
    tensorboard_path = os.path.join(save_path, 'lightning_logs')
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(tensorboard_path)
    model = timm.create_model('efficientnetv2_l', in_chans=3, num_classes=6).cuda()
    renderer = render_init()
    val_renderer = render_init('val')
    seg = None
    file_path = '/mnt/share/shenfeihong/data/smile_/C01001659595'
    upper, lower, mid = load_up_low(file_path, mode='T2', num_teeth=1)
    upper = meshes_to_tensor(upper).cuda()
    lower = meshes_to_tensor(lower).cuda()
    optimize = optim.AdamW(model.parameters(), lr = 0.0001)

    dl = get_loader_unet(size=14, mode='train')
    val_dl = get_loader_unet(size=1, mode='val')
    # iteration = iter(dl)
    for index in range(50):
        if not index % 5:
            torch.save(model.state_dict(), f"{save_path}/{index}.pt")
        for idx,batch in enumerate(dl):

            optimize.zero_grad()
            mouth = batch['mouth'].cuda()
            mask = batch['mask'].cuda()
            upper_tid_seg = batch['label_up'].cuda()
            lower_tid_seg = batch['label_down'].cuda()
            
            up_edge = batch['up_edge'].cuda()
            down_edge = batch["down_edge"].cuda()
            
            exist = batch['lower'].cuda().unsqueeze(-1)
            
            zero_ = torch.tensor([0]).unsqueeze(0).repeat(mask.shape[0],1).cuda()

            output = model(mouth*mask-up_edge-down_edge)
            dist_up = torch.cat((zero_,zero_,output[:,4:5]),1)
            dist_down = torch.cat((zero_,zero_,output[:,3:4]*exist),1)
            angle = torch.cat((zero_-1.396,zero_,zero_),1)
            movement = torch.cat((output[:,0:1],output[:,1:2],output[:,2:3]*10+470),1)

            
            deepmap_upper, deepmap_lower = render_(renderer, upper, lower, mid, mask, angle, movement, [dist_up,dist_down])

            loss = F.mse_loss(upper_tid_seg, deepmap_upper)+0.1*F.mse_loss(lower_tid_seg, deepmap_lower)
            loss.backward()
            optimize.step()
            writer.add_scalar('Loss/train', loss.item(), idx)
            writer.add_image('Image/Input', torch.clip((mouth[0]+1)/2,min=0,max=1), idx)
            gap = torch.zeros((3,256,256))
            gap[0] = upper_tid_seg[0][0]
            gap[1] = deepmap_upper[0][0]
            writer.add_image('Image/Gap', gap, idx)
            gap[0] = lower_tid_seg[0][0]
            gap[1] = deepmap_lower[0][0]
            writer.add_image('Image/GapDown', gap, idx)
            
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
    
    train()