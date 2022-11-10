from train_encoder import PSP
from PIL import Image
import numpy as np
import torch
from stylegan2.dataset import *
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from test import Yolo, Segmentation, sigmoid
from utils import loose_bbox

ckpt_encoder = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.8/encoder_ckpt/3.pkl'
ckpt_decoder = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.8/040000.pt'
psp = PSP(256,512,8).cuda()
ckpt_decoder_ = torch.load(ckpt_decoder, map_location=lambda storage, loc: storage)
ckpt_encoder_ = torch.load(ckpt_encoder, map_location=lambda storage, loc: storage)
psp.decoder.load_state_dict(ckpt_decoder_["g_ema"])
psp.psp_encoder.load_state_dict(ckpt_encoder_)
yolo = Yolo('/mnt/share/shenfeihong/weight/pretrain/yolo.onnx', (640, 640))
seg = Segmentation('/mnt/share/shenfeihong/weight/pretrain/edge.onnx', (256, 256))
save_path = '/mnt/share/shenfeihong/data/test/11.9.2022.res'
def test_single_full(img_path):


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
    mask = cv2.dilate(mask, kernel=np.ones((30, 30)))-cv2.dilate(mask, kernel=np.ones((3, 3)))
    mask = torch.from_numpy(mask.astype(np.float32)[None][None]).cuda()
    mouth = mouth/255*2-1
    mouth = torch.from_numpy(mouth.transpose(2,0,1).astype(np.float32)[None]).cuda()
    print(mask.shape, mouth.shape)
    img,_ = psp(mouth, mask, mix=True)
    utils.save_image(
        img,
        f"{save_path}/{img_path.split('/')[-1]}",
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )
import os
path = '/mnt/share/shenfeihong/data/test/11.8.2022'
for file in os.listdir(path):
    test_single_full(os.path.join(path, file))
    # break
    