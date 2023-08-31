import random
import os

import numpy as np
import torch
from torch.utils import data
import cv2

from mydataset import SmileDataset


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def train(loader, generator, discriminator, g_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)
    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        real_batch = next(loader)
        real_img = real_batch['img'].to(device)
        bg = real_batch['bg'].to(device)
        input_semantic = real_batch['input_semantic'].to(device)
        mask = real_batch['mask'].to(device)

        # real_img = real_img.to(device)

        with torch.no_grad():
            g_ema.eval()
            # print(len().parameters())
            # sample, _ = g_ema([sample_z])
            for _ in range(10):
                noise = mixing_noise(args.batch, args.latent, 0., device)
                sample, _ = g_ema(noise, geometry=input_semantic, background=bg, mask=mask)

                from torchvision.utils import make_grid
                import cv2

                grid = make_grid(torch.cat([sample, real_img, input_semantic[:, :3], input_semantic[:, -3:]], dim=0),
                                 nrow=4, normalize=True, range=(-1, 1))
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()



from PIL import Image


def inference(generator, case_dir, device='cuda', latent=512, mixing=0., num_loop=1):
    mouth = np.array(Image.open(os.path.join(case_dir, 'mouth.jpg')))
    fg = np.array(Image.open(os.path.join(case_dir, 'MouthMask.png')))
    fg = fg.astype(np.float32) / 255

    mask = cv2.dilate(fg, np.ones((7, 7)))

    dilated_fg = cv2.dilate(fg, np.ones((5,5)))
    # fg = mask
    transition = mask - dilated_fg
    mask = mask[..., None]

    img = mouth.astype(np.float32) / 255 * 2 - 1
    bg = img * (1 - mask)
    semantic = np.zeros((5, 256, 256), dtype=np.float32)
    # semantic[5] = transition

    step_dir = os.path.join(case_dir, 'steps')
    steps = [f for f in os.listdir(step_dir) if f.startswith('mask')]

    noise = mixing_noise(1, latent, mixing, device)
    for mask_file in [steps[-1]]:
        teeth_mask = np.array(Image.open(os.path.join(step_dir, mask_file)).convert('L'))
        # teeth_mask = teeth_mask.astype(np.float32) / 255
        teeth_mask[teeth_mask == teeth_mask.max()] = 0
        teeth_mask[teeth_mask > 0] = 1
        teeth_mask = teeth_mask.astype(np.float32)

        # teeth_mask = cv2.dilate(teeth_mask, np.ones((ksize, ksize)))

        edge_mask = cv2.imread(os.path.join(step_dir, mask_file.replace('mask_', '')))

        up_teeth = edge_mask[..., 2].astype(np.float32) / 255
        # up_teeth = cv2.dilate(up_teeth, np.ones((3, 3)))
        down_teeth = edge_mask[..., 1].astype(np.float32) / 255
        # down_teeth = cv2.dilate(down_teeth, np.ones((3, 3)))

        edge = np.logical_or(up_teeth, down_teeth).astype(np.float32)
        # semantic[0] = dilated_fg - teeth_mask * fg
        semantic[0] = mask.squeeze()
        semantic[1] = edge
        semantic[2] = teeth_mask * fg
        semantic[3] = up_teeth
        semantic[4] = down_teeth

        teeth = img*mask
        with torch.no_grad():

            input_semantic = torch.tensor(semantic[None], device=device)
            input_bg = torch.tensor(bg.transpose(2,0,1)[None], device=device)
            input_mask = torch.tensor(mask.transpose(2,0,1)[None], device=device)
            input_teeth = torch.tensor(teeth.transpose(2,0,1)[None], device=device)

            print(input_bg.max(), input_bg.min())
            print(input_mask.max(), input_mask.min())

            input_dict =(noise,{
                # 'styles': noise,
                'geometry': input_semantic,
                'background':input_bg,
            })
            input_dict = (noise,input_semantic, input_bg, input_mask,
                          False, None, 1, None, False, None, False)

            # torch.onnx.export(g_ema, input_dict, 'stylegan.onnx', opset_version=12)
            # exit()
            for i in range(num_loop):
                sample, _ = generator(input_teeth, geometry=input_semantic, background=input_bg,
                                      mask=input_mask,
                                      randomize_noise=False)

                sample = sample.cpu().numpy()[0]
                sample = (sample + 1) / 2


        # real_img = real_img.to(device)


if __name__ == "__main__":
    from cgan import Condition_Generator

    device = "cuda"
    # ckpt = r'E:\stylegan2-pytorch\checkpoint\non\160000.pt'
    ckpt = r'checkpoint/085000.pt'
    size = 256
    latent = 512
    n_mlp = 8

    # from model import Discriminator

    generator = Condition_Generator(size, latent, n_mlp, ).to(device)

    g_ema = Condition_Generator(size, latent, n_mlp, ).to(device)
    g_ema.eval()

    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    # generator.load_state_dict(ckpt["g"])
    g_ema.load_state_dict(ckpt["g_ema"])

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in generator.parameters())))

    path = 'C:\data\smile_synthesis\smile_segmap'
    dataset = SmileDataset(path, None, size)
    loader = data.DataLoader(
        dataset,
        batch_size=4,
        # sampler=data_sampler(dataset, shuffle=True, distributed=False),

        drop_last=True,
    )

    from encoder_train import PSPModel
    # pl_model = PSPModel(ckpt=r'checkpoint/085000.pt')
    ckpt = r'weights/encoder/version_0/checkpoints/global_steps=00-val_loss=0.020685.ckpt'
    state_dict = torch.load(ckpt)

    model = PSPModel.load_from_checkpoint(
        r'weights/encoder/version_0/checkpoints/global_steps=00-val_loss=0.020685.ckpt',
        ckpt=r'checkpoint/085000.pt')
    model.to(device)

    data_dir = r'E:\fitting\data\test\test_05_12'
    for case in os.listdir(data_dir):
        case_dir = os.path.join(data_dir, case)
        inference(model.net, case_dir, )
