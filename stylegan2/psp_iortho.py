import math
import torch
from torch import nn
import sys
sys.path.append('.')
sys.path.append('..')
from encoders import psp_encoders
from stylegan2.model_ex import Generator

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

class pSp(nn.Module):
    def __init__(self, opts=None):
        super(pSp, self).__init__()
        # compute number of style inputs based on the output resolution
        self.n_styles = int(math.log(256, 2)) * 2 - 2
        # Define architecture
        self.encoder1 = psp_encoders.GradualStyleEncoder(50, 'ir_se', input_nc=opts.input_nc, use_skip=opts.use_skip)
        self.encoder2 = psp_encoders.GradualStyleEncoder(50, 'ir_se', input_nc=opts.input_nc, use_skip=opts.use_skip)
        self.decoder = Generator(256, 512, 8)
        if opts.stylegan_weights is not None:
            ckpt = torch.load(opts.stylegan_weights, map_location='cpu')
            self.decoder.load_state_dict(ckpt['g_ema'])
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.latent_avg = None
        self.start_from_latent_avg = opts.start_from_latent_avg
        print('using psp iortho')
    # x1: image for first-layer feature f. 
    # x2: image for style latent code w+. If not specified, x2=x1.
    # inject_latent: for sketch/mask-to-face translation, another latent code to fuse with w+
    # latent_mask: fuse w+ and inject_latent with the mask (1~7 use w+ and 8~18 use inject_latent)
    # use_feature: use f. Otherwise, use the orginal StyleGAN first-layer constant 4*4 feature 
    # first_layer_feature_ind: always=0, means the 1st layer of G accept f
    # use_skip: use skip connection.
    # zero_noise: use zero noises. 
    # editing_w: the editing vector v for video face editing
    def forward(self, x1, x2=None, resize=True, latent_mask=None, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, use_feature=True, 
                first_layer_feature_ind=0, use_skip=False, zero_noise=False, editing_w=None): ##### modified
        
        feats = None # f and the skipped encoder features
        codes, feats = self.encoder1(x1, return_feat=True, return_full=use_skip) ##### modified

        codes, feats2 = self.encoder2(x2, return_feat=True, return_full=use_skip) ##### modified

        # E_W^{1:7}(T(x1)) concatenate E_W^{8:18}(w~)
        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0
        if self.start_from_latent_avg:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)   
        first_layer_feats, skip_layer_feats, fusion = None, None, None ##### modified            
        if use_feature: ##### modified
            first_layer_feats = feats[0:2]+feats2[0:2] # use f
        if use_skip: ##### modified
            skip_layer_feats = feats[2:] # use skipped encoder feature
            fusion = self.encoder1.fusion # use fusion layer to fuse encoder feature and decoder feature.
            
        images, result_latent = self.decoder([codes],
                                             input_is_latent=True,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents,
                                             first_layer_feature=first_layer_feats,
                                             first_layer_feature_ind=first_layer_feature_ind,
                                             skip_layer_feature=skip_layer_feats,
                                             fusion_block=fusion,
                                             zero_noise=zero_noise,
                                             editing_w=editing_w) ##### modified

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images
