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
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.state import PartialState

from utils import common, train_utils
from criteria import id_loss, w_norm
# from ex_dataset import ImagesDataset, TestDataset
from mydataset import YangOldNew, GeneratedDepth
from criteria.lpips.lpips import LPIPS
from stylegan2.model import Discriminator ##### modified
from script.ranger import Ranger

def replace_batchnorm(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_batchnorm(module)
        
        if isinstance(module, torch.nn.BatchNorm2d):
            model._modules[name] = torch.nn.GroupNorm(32, module.num_features)

    return model 

class MyObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)
            
class Coach:
    def __init__(self, opts, network='unet'):
        self.opts = opts
        self.global_step = 0

        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)
        
        # Initialize network
        if network=='unet':
            from stylegan2.model_cond import pSp
        else:
            from stylegan2.psp_iortho import pSp
        net = pSp(self.opts)
        net = replace_batchnorm(net)
        # if self.opts.stylegan_weights is not None:
        #     ckpt = torch.load(self.opts.stylegan_weights, map_location='cpu')
        #     net.load_state_dict(ckpt['g_ema'], strict=True)
        if self.opts.adv_lambda > 0:  ##### modified, add discriminator
            discriminator = Discriminator(256, channel_multiplier=2)
            if self.opts.stylegan_weights is not None:
                ckpt = torch.load(self.opts.stylegan_weights, map_location='cpu')
                discriminator.load_state_dict(ckpt['d'], strict=False)
            discriminator = discriminator
            discriminator_optimizer = torch.optim.Adam(list(discriminator.parameters()),
                                                            lr=self.opts.learning_rate)
            
        # Estimate latent_avg via dense sampling if latent_avg is not available
        # if net.latent_avg is None:
        #     net.latent_avg = net.decoder.mean_latent(int(1e5))[0].detach()

        # Initialize loss
        if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
            raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')
            
        # Initialize optimizer
        optimizer = self.configure_optimizers(net)

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = None
        self.latent_mask = None
        self.editing_w = None
        self.directions = None
        

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[kwargs])
        self.state = PartialState()
        
        optimizer, net, train_dataloader, test_dataloader = self.accelerator.prepare(optimizer, net, train_dataloader, test_dataloader)
        self.optimizer = optimizer
        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        if self.opts.adv_lambda > 0:  ##### modified, add discriminator
            discriminator, discriminator_optimizer = self.accelerator.prepare(discriminator, discriminator_optimizer)
            self.discriminator = discriminator
            self.discriminator_optimizer = discriminator_optimizer
        self.mse_loss = nn.MSELoss().eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = self.accelerator.prepare(LPIPS(net_type='alex'))
        if self.opts.id_lambda > 0:
            self.id_loss = self.accelerator.prepare(id_loss.IDLoss())
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                
                #************************ Data Preparation **************************
                
                cond_img = batch['cond']
                real_img = batch['images']


                y_hat = self.net.forward(cond_img, return_latents=False)      
                
                # adversarial loss
                if self.opts.adv_lambda > 0: 
                    d_loss_dict = self.train_discriminator(real_img, y_hat)
                
                # calculate losses
                loss, loss_dict, id_logs = self.calc_loss(cond_img[:,:3,:,:], real_img, y_hat)
                
                if self.opts.adv_lambda > 0:
                    loss_dict = {**d_loss_dict, **loss_dict}
                
                self.accelerator.backward(loss)
                self.optimizer.step()

                #************************ logging and saving model************************** 
                
                    
                if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(cond_img[:,:3,:,:], real_img, cond_img[:,-3:,:,:], y_hat, title='train')
                    
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            cond_img = batch['cond']
            real_img = batch['images']

            with torch.no_grad():
                y_hat = self.net.forward(cond_img, return_latents=False)                  
            
            self.parse_and_log_images(cond_img[:,:3,:,:], real_img, cond_img[:,-3:,:,:], y_hat,
                                      title='test',
                                      subscript='{:04d}'.format(batch_idx))

        self.net.train()
        return None

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        self.accelerator.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def configure_optimizers(self, net):
        params = list(net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(net.decoder.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        # train_dataset = YangOldNew('train')
        # test_dataset = YangOldNew('test')
        train_dataset = GeneratedDepth(args.data_path, 'train')
        test_dataset = GeneratedDepth(args.data_path, 'test')
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, y0_hat=None):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.lpips_lambda_crop > 0:
            loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
            loss += loss_lpips_crop * self.opts.lpips_lambda_crop
        if self.opts.l2_lambda_crop > 0:
            loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_l2_crop'] = float(loss_l2_crop)
            loss += loss_l2_crop * self.opts.l2_lambda_crop
        # if self.opts.w_norm_lambda > 0: 
        #     loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
        #     loss_dict['loss_w_norm'] = float(loss_w_norm)
        #     loss += loss_w_norm * self.opts.w_norm_lambda
        if self.opts.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.opts.moco_lambda
        if self.opts.adv_lambda > 0:  ##### modified
            loss_g = F.softplus(-self.discriminator(y_hat)).mean()
            loss_dict['loss_g'] = float(loss_g)
            loss += loss_g * self.opts.adv_lambda
        if self.opts.tmp_lambda > 0 and y0_hat is not None:  ##### modified
            loss_tmp = ((y_hat-y0_hat)**2).mean()
            loss_dict['loss_tmp'] = float(loss_tmp)
            loss += loss_tmp * self.opts.tmp_lambda * min(1, 4.0*(self.global_step/self.opts.max_steps-0.5))
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, x, y, z, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'cond_face': common.tensor2im(z[i]),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{str(step).zfill(6)}_{subscript}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{str(step).zfill(6)}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.net)
        save_dict = {
            'state_dict': unwrapped_model.state_dict(),
            'opts': vars(self.opts)
        }
        if self.opts.adv_lambda > 0:  ##### modified
            unwrapped_discriminator = self.accelerator.unwrap_model(self.net)
            save_dict['discriminator'] = unwrapped_discriminator.state_dict()
        return save_dict             
            

    ##### modified
    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['loss_d_real'] = float(real_loss)
        loss_dict['loss_d_fake'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, real_img, fake_img):
        loss_dict = {}
        self.requires_grad(self.discriminator, True)
        self.discriminator_optimizer.zero_grad()

        real_pred = self.discriminator(real_img)
        fake_pred = self.discriminator(fake_img.detach())
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['loss_d'] = float(loss)
        loss = loss * self.opts.adv_lambda 

        self.accelerator.backward(loss)
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            self.discriminator_optimizer.zero_grad()
            real_img = real_img.detach()
            real_img.requires_grad = True
            real_pred = self.discriminator(real_img)
            r1_loss = self.discriminator_r1_loss(real_pred, real_img)

            self.discriminator.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
            self.accelerator.backward(r1_final_loss)
            self.discriminator_optimizer.step()
            loss_dict['loss_r1'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict
    
    def validate_discriminator(self, real_img, fake_img):
        with torch.no_grad():
            loss_dict = {}
            real_pred = self.discriminator(real_img)
            fake_pred = self.discriminator(fake_img.detach())
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['loss_d'] = float(loss)
            loss = loss * self.opts.adv_lambda 
            return loss_dict

if __name__=='__main__':
    opts = ArgumentParser()
    opts.add_argument('--exp_dir', default='/ssd/gregory/smile/iortho/',type=str, help='Path to experiment output directory')
    opts.add_argument('--dataset_type', default='ffhq_sketch_to_face', type=str, help='Type of dataset/experiment to run')
    opts.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use') 
    opts.add_argument('--input_nc', default=2, type=int, help='Number of input image channels to the psp encoder')
    opts.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
    opts.add_argument('--output_size', default=1024, type=int, help='Output size of generator')

    # new options for StyleGANEX
    opts.add_argument('--feat_ind', default=0, type=int, help='Layer index of G to accept the first-layer feature')
    opts.add_argument('--max_pooling', action="store_true", help='Apply max pooling or average pooling')
    opts.add_argument('--use_skip', default=True, help='Using skip connection from the encoder to the styleconv layers of G')
    opts.add_argument('--use_feature', default=True, help='Using skip connection from the encoder to the styleconv layers of G')
    opts.add_argument('--use_skip_torgb', default=False, help='Using skip connection from the encoder to the toRGB layers of G.')
    opts.add_argument('--skip_max_layer', default=7, type=int, help='Layer used for skip connection. 1,2,3,4,5,6,7 correspond to 4,8,16,32,64,128,256')
    opts.add_argument('--crop_face', action="store_true", help='Use aligned cropped face to predict style latent code w+')
    opts.add_argument('--affine_augment', action="store_true", help='Apply random affine transformation during training')
    opts.add_argument('--random_crop', action="store_true", help='Apply random crop during training')
    # for SR
    opts.add_argument('--resize_factors', type=str, default=None, help='For super-res, comma-separated resize factors to use for inference.')
    opts.add_argument('--blind_sr', action="store_true", help='Whether training blind SR (will use ./datasetsffhq_degradation_dataset.py)')  
    # for sketch/mask to face translation
    opts.add_argument('--use_latent_mask', action="store_true", help='For segmentation/sketch to face translation, fuse w+ from two sources')
    opts.add_argument('--latent_mask', type=str, default='8,9,10,11,12,13,14,15,16,17', help='Comma-separated list of latents to perform style-mixing with')
    opts.add_argument('--res_num', default=2, type=int, help='Layer number of the resblocks of the translation network T')        
    # for video face toonify
    opts.add_argument('--toonify_weights', default=None, type=str, help='Path to Toonify StyleGAN model weights')
    # for video face editing
    opts.add_argument('--generate_training_data', action="store_true", help='Whether generating training data (for video editing) or load real data')
    opts.add_argument('--use_att', default=0, type=int, help='Layer of MLP used for attention, 0 not use attention')
    opts.add_argument('--editing_w_path', type=str, default=None, help='Path to the editing vector v')
    opts.add_argument('--zero_noise', action="store_true", help='Whether using zero noises')
    opts.add_argument('--direction_path', type=str, default=None, help='Path to the direction vector to augment generated data')

    opts.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    opts.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
    opts.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
    opts.add_argument('--test_workers', default=8, type=int, help='Number of test/inference dataloader workers')

    opts.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
    opts.add_argument('--optim_name', default='adam', type=str, help='Which optimizer to use')
    opts.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
    opts.add_argument('--start_from_latent_avg', default=False, help='Whether to add average latent vector to generate codes from encoder.')
    opts.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')

    opts.add_argument('--lpips_lambda', default=1.0, type=float, help='LPIPS loss multiplier factor')
    opts.add_argument('--id_lambda', default=0, type=float, help='ID loss multiplier factor')
    opts.add_argument('--l2_lambda', default=0.1, type=float, help='L2 loss multiplier factor')
    opts.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
    opts.add_argument('--lpips_lambda_crop', default=0, type=float, help='LPIPS loss multiplier factor for inner image region')
    opts.add_argument('--l2_lambda_crop', default=0, type=float, help='L2 loss multiplier factor for inner image region')
    opts.add_argument('--moco_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
    opts.add_argument('--adv_lambda', default=1, type=float, help='Adversarial loss multiplier factor')
    opts.add_argument('--d_reg_every', default=16, type=int, help='Interval of the applying r1 regularization')
    opts.add_argument('--r1', default=1, type=float, help="weight of the r1 regularization")
    opts.add_argument('--tmp_lambda', default=0, type=float, help='Temporal loss multiplier factor')

    opts.add_argument('--stylegan_weights', default='/ssd/gregory/smile/ori_style/checkpoint/150000.pt', type=str, help='Path to StyleGAN model weights')
    # opts.add_argument('--decoder_path', default='/ssd/gregory/smile/ori_style/checkpoint/150000.pt', type=str, help='Path to pSp model checkpoint')
    # opts.add_argument('--discriminator_path', default=None, type=str, help='Path to pSp model checkpoint')

    opts.add_argument('--max_steps', default=250000, type=int, help='Maximum number of training steps')
    opts.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
    opts.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
    opts.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
    opts.add_argument('--save_interval', default=10000, type=int, help='Model checkpoint interval')

    # arguments for weights & biases support
    opts.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')


    args = opts.parse_args()
    args.exp_dir = '/ssd/gregory/smile/orthovis/10.8'
    args.stylegan_weights = '/ssd/gregory/smile/ori_style/checkpoint/150000.pt'
    args.data_path = '/ssd/gregory/smile/out/'
    args.exp_dir = '/data/shenfeihong/smile/orthovis/10.8'
    args.stylegan_weights = '/data/shenfeihong/smile/ori_style/checkpoint/ori_style_150000.pt'
    args.data_path = '/data/shenfeihong/smile/out/'
    
    print(args)
    coach = Coach(args)
    coach.train()
    # test()