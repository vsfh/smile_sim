import os
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys
sys.path.append('.')
sys.path.append('..')
matplotlib.use('Agg')

import torch
from torch import nn, autograd  ##### modified
from torch.utils.data import DataLoader

import numpy as np

# from ex_dataset import ImagesDataset, TestDataset
from ex_dataset import MyTestDataset


class MyObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def test():
    import cv2
    ckpt = torch.load('/mnt/e/paper/smile/weight/orthovis_iteration_200000.pt')
    opts = MyObject(ckpt['opts'])
    opts.stylegan_weights = '/mnt/e/paper/smile/weight/ori_style_150000.pt'

    from stylegan2.model_cond import pSp

    net = pSp(opts).cuda()
    print(ckpt['opts'])
    net = pSp(opts=opts).cuda()
    net.load_state_dict(ckpt['state_dict'])
    # net.eval()
    test_loader = DataLoader(MyTestDataset('train'),
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=int(opts.test_workers),
                                    drop_last=True)
    for batch_idx, batch in enumerate(test_loader):
        cond_img = batch['cond'].cuda().float()
        real_img = batch['images'].cuda().float()

        with torch.no_grad():
            y_hat, latent = net.forward(cond_img, return_latents=True)   
            y_hat = y_hat*cond_img[:,2:3,:,:]+real_img*(1-cond_img[:,2:3,:,:])
            
        im = y_hat[0].detach().cpu().numpy().clip(0,1)
        im = im.transpose(1,2,0)*255
        im = im.astype(np.uint8)[...,::-1]
        cv2.imwrite(f'/mnt/e/paper/smile/orthovis_res/{str(batch_idx).zfill(5)}.png', im)
if __name__=='__main__':
    test()