from cgan import TeethGenerator
from encoders.psp_encoders import GradualStyleEncoder
import torch
import torch.nn as nn

class onnxmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.psp_encoder = GradualStyleEncoder(50, 'ir_se')
        self.decoder = TeethGenerator(256, 512, n_mlp=8)
        self.sample_z = torch.randn((1,512)).cuda()*0.12

    def forward(self, real_img, mask, big_mask):
        
        codes = self.psp_encoder(real_img)
        sample_z = self.decoder.style(self.sample_z)
        codes = [codes,sample_z]
        images, result_latent = self.decoder(codes, real_image=real_img, mask=mask,
                                             input_is_latent=True,
                                             randomize_noise=False)
        images = real_img*(1-big_mask)+images*big_mask
        return images

class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        # self.psp_encoder = GradualStyleEncoder(50, 'ir_se')
        self.decoder = TeethGenerator(256, 512, n_mlp=8)
        self.sample_z = torch.load('/mnt/share/shenfeihong/weight/smile-sim/2022.11.11/test/_3.pth').cuda()
        # self.sample_z = torch.randn((1,512)).cuda()

    def forward(self, real_img, mask, big_mask):
        
        sample_z = self.decoder.style(self.sample_z)
        codes = [sample_z]
        images,_ = self.decoder(codes, real_image=real_img, mask=mask,
                                            input_is_latent=True,
                                            randomize_noise=False)
        images = real_img*(1-big_mask)+images*big_mask
        return images
       
def convert_to_onnx():
    dynamic_axes = {
        'input_image': {0: 'batch_size'},
        'mask': {0: 'batch_size'},
        'big_mask': {0: 'batch_size'},
        'align_img': {0: 'batch_size'}
    }

    output_path = 'model_single.onnx'
    # input = './smile/C01001459133.jpg'
    input1 = torch.randn(1, 3, 256, 256).cuda()
    input2 = torch.randn(1, 1, 256, 256).cuda()
    input3 = torch.randn(1, 1, 256, 256).cuda()
    model = Gen().eval().cuda()
    ckpt_encoder = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.8/encoder_ckpt/3.pkl'
    ckpt_decoder = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.11/040000.pt'
    ckpt_decoder_ = torch.load(ckpt_decoder, map_location=lambda storage, loc: storage)
    # ckpt_encoder_ = torch.load(ckpt_encoder, map_location=lambda storage, loc: storage)
    model.decoder.load_state_dict(ckpt_decoder_["g_ema"])
    # model.psp_encoder.load_state_dict(ckpt_encoder_)
    input_name = ['input_image','mask','big_mask']
    output_name = ['align_img']
    torch.onnx.export(model, (input1, input2, input3), output_path, export_params=True, input_names=input_name, output_names=output_name,
                      opset_version=13, dynamic_axes=dynamic_axes)
import cv2
import numpy as np
import onnxruntime
import os
def onnx_infer():
    from test import Yolo, Segmentation, sigmoid
    from utils import loose_bbox
    
    yolo = Yolo('/mnt/share/shenfeihong/weight/pretrain/yolo.onnx', (640, 640))
    seg = Segmentation('/mnt/share/shenfeihong/weight/pretrain/edge.onnx', (256, 256))
    sample_dir = '/mnt/share/shenfeihong/data/test/11.8.2022'
    save_path = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.11/onnx'
    sess = onnxruntime.InferenceSession('model_single.onnx',
                                        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                'CPUExecutionProvider'])   
    for file in os.listdir(sample_dir):
        img_path = os.path.join(sample_dir, file)
        image = cv2.imread(img_path)
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        height, width = image.shape[:2]

        objs = yolo.predict_with_argmax(image, show=False)

        mouth_objs = objs[2]
        x1, y1, x2, y2 = mouth_objs

        w, h = (x2 - x1), (y2 - y1)

        half = max(w, h) * 1.1 / 2

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
        x1, y1, x2, y2 = loose_bbox([x1, y1, x2, y2], (width, height))
        x, y = int(x1 * 128 / half), int(y1 * 128 / half) + 2

        image = cv2.resize(image, (int(width * 128 / half), int(height * 128 / half)), cv2.INTER_AREA)
        mouth = image[y: y + 256, x: x + 256]
        result = seg.predict(mouth)

        mask = (result[..., 0] > 0.6).astype(np.float32)
        big_mask = cv2.dilate(mask, kernel=np.ones((3, 3)))
        mask = cv2.dilate(mask, kernel=np.ones((33, 33)))-big_mask

        mouth_tensor = mouth/255*2-1
        mouth_tensor = mouth_tensor.transpose(2,0,1).astype(np.float32)[None]

        mask = mask.astype(np.float32)[None][None]
        big_mask = big_mask.astype(np.float32)[None][None]

        align_img = sess.run([], {'input_image':mouth_tensor,'mask':mask,'big_mask':big_mask}) 
        align_img = align_img[0][0].transpose(1,2,0)*255/2+255/2
        image[y: y + 256, x: x + 256] = align_img.clip(0,255)
        cv2.imwrite(f'{save_path}/{file}', cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
        # print(align_img[0].shape, len(align_img))
    
def model_infer():
    model = onnxmodel().eval().cuda()
    ckpt_encoder = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.8/encoder_ckpt/3.pkl'
    ckpt_decoder = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.8/040000.pt'
    ckpt_decoder_ = torch.load(ckpt_decoder, map_location=lambda storage, loc: storage)
    ckpt_encoder_ = torch.load(ckpt_encoder, map_location=lambda storage, loc: storage)
    model.decoder.load_state_dict(ckpt_decoder_["g_ema"])
    model.psp_encoder.load_state_dict(ckpt_encoder_)    
    
    img = cv2.imread('/mnt/share/shenfeihong/data/test/zyy/mouth.jpg')
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))/255*2-1
    img = img.transpose(2,0,1).astype(np.float32)[None]
    
    mask = cv2.imread('/mnt/share/shenfeihong/data/test/zyy/MouthMask.png')
    mask = np.array(mask)/255
    if len(mask.shape) == 3:
        mask = mask[...,0]
        
    big_mask = cv2.dilate(mask, kernel=np.ones((2, 2)))
    mask = cv2.dilate(mask, kernel=np.ones((30, 30)))-big_mask

    mask = mask.astype(np.float32)[None][None]
    big_mask = big_mask.astype(np.float32)[None][None]
    
    img = torch.from_numpy(img).cuda()
    mask = torch.from_numpy(mask).cuda()
    big_mask = torch.from_numpy(big_mask).cuda()
    
    align_img = model(img, mask, big_mask)
    align_img = align_img[0].detach().cpu().numpy().transpose(1,2,0)*255/2+255/2
    cv2.imwrite('img.jpg', cv2.cvtColor(align_img, cv2.COLOR_RGB2BGR).astype(np.uint8))
convert_to_onnx()