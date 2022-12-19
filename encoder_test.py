from encoder_train import PSP
from PIL import Image
import numpy as np
import torch
from stylegan2.dataset import *
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from test import Yolo, Segmentation, sigmoid
from utils import loose_bbox
import onnxruntime
sess = onnxruntime.InferenceSession('model_single.onnx',
                                    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                            'CPUExecutionProvider']) 
yolo = Yolo('/mnt/share/shenfeihong/weight/pretrain/yolo.onnx', (640, 640))
seg = Segmentation('/mnt/share/shenfeihong/weight/pretrain/edge.onnx', (256, 256))


ckpt_encoder = './2022.12.13/encoder/ckpt/13.pt'
ckpt = '/mnt/share/shenfeihong/tmp/wo_edge/040000.pt'

pl_model = PSP(ckpt=ckpt).cuda()
ckpt_encoder_ = torch.load(ckpt_encoder, map_location=lambda storage, loc: storage)
pl_model.psp_encoder.load_state_dict(ckpt_encoder_)
def test_single_full(img_path, save_path):

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
    mask = cv2.dilate(mask, kernel=np.ones((23, 23)))-cv2.dilate(mask, kernel=np.ones((3, 3)))
    mask = torch.from_numpy(mask.astype(np.float32)[None][None]).cuda()
    mouth = mouth/255*2-1
    mouth = torch.from_numpy(mouth.transpose(2,0,1).astype(np.float32)[None]).cuda()
    print(mask.shape, mouth.shape)
    img,_ = pl_model(mouth, mask, mix=True)
    utils.save_image(
        img,
        f"{save_path}/{img_path.split('/')[-1]}",
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )
def onnx_test(img_path):
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
    # big_mask = cv2.dilate(mask, kernel=np.ones((2, 2)))
    big_mask = mask
    mask = cv2.dilate(mask, kernel=np.ones((30, 30)))-big_mask

    mask = mask.astype(np.float32)[None][None]
    big_mask = big_mask.astype(np.float32)[None][None]  
    img = mouth/255*2-1
    img = img.transpose(2,0,1).astype(np.float32)[None]
  
    align_img = sess.run([], {'input_image':img,'mask':mask,'big_mask':big_mask}) 
    align_img = align_img[0][0].transpose(1,2,0)*255/2+255/2
    image[y: y + 256, x: x + 256] = align_img.clip(0,255)
    cv2.imwrite(f"{save_path}/{img_path.split('/')[-1]}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
    
import os
sample_dir = '/mnt/share/shenfeihong/tmp/test/40photo'
save_path = './2022.12.13/encoder/test'
for file in os.listdir(sample_dir):
    test_single_full(os.path.join(sample_dir, file), save_path)
    # break
    