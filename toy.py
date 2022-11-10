from PIL import Image
import numpy as np
import torch
from stylegan2.dataset import *
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils

def noise():
    from train_less_cgan import mixing_noise
    from cgan import TeethGenerator
    model = TeethGenerator(256, 512, 8).cuda()
    ckpt_down = torch.load('/mnt/share/shenfeihong/weight/smile-sim/2022.11.8/050000.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt_down["g_ema"])
    # sample_z = torch.randn((4,512)).cuda()
    
    sample_dir = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.8/test'
    dataset = fuck(mode='encoder')
    loader = data.DataLoader(
        dataset,
        batch_size=6,
        sampler=data.RandomSampler(dataset),
        drop_last=True,
    )    
    sample_z = torch.randn((6,512)).cuda()*0.12
    iteration = iter(loader)
    for i in range(100):
        batch = next(iteration)
        real_img = batch['images'].cuda()
        mask = batch['mask'].cuda()
        # code = [model.style(s) for s in [sample_z]]
        # print(code[0].shape)
        sample, _ = model([sample_z], real_image=real_img, mask=mask)
        utils.save_image(
            sample,
            f"{sample_dir}/{i}.png",
            nrow=2,
            normalize=True,
            range=(-1, 1),
        )
if __name__ == "__main__":
    noise()
    a = 1