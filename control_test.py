from control_gan import ControlModel
import torch
import cv2
import numpy as np
import os
from torchvision import transforms, utils
def preprocess(img):
    img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    if len(img_resize.shape)==2:
        img_resize = img_resize[:,:,None].repeat(1,1,3)
    im = np.ascontiguousarray(img_resize.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
    im = torch.from_numpy(im)  # to torch
    im = im.float()  # uint8 to fp16/32
    im /= 255.0  # 0-255 to 0.0-1.0
    return im
    
def get_cond():
    path = '/mnt/share/shenfeihong/data/smile_to_b_test/x'
    path = '/mnt/share/shenfeihong/data/TrainDataNew/Teeth/C01003236374'
    img = cv2.imread(os.path.join(path, 'Img.jpg'))
    mask = cv2.imread(os.path.join(path, 'MouthMask.png'))
    edge = cv2.imread(os.path.join(path, 'TeethEdge.png'))
    tmask = cv2.imread(os.path.join(path, 'TeethMasks.png'))
    # tmask[tmask==tmask.max()] = 0
    # tmask[tmask>0] = 255
    # tmask[mask==0] = 0
    # edge = cv2.dilate(edge, kernel=np.ones((3,3))).sum(-1).clip(0,255)[...,None].repeat(3,2)
    
    im = preprocess(img)
    mk = preprocess(mask)
    ed = preprocess(edge)
    tk = preprocess(tmask)
    
    cond = mk*ed*0.1 + (1-mk)*im + (1-ed)*tk
    return cond
    
def get_cond_x():
    path = '/mnt/share/shenfeihong/data/smile_to_b_test/x'
    # path = '/mnt/share/shenfeihong/data/TrainDataNew/Teeth/C01003236374'
    img = cv2.imread(os.path.join(path, 'mouth.jpg'))
    mask = cv2.imread(os.path.join(path, 'MouthMask.png'))
    edge = cv2.imread(os.path.join(path, 'steps','step_025.png'))
    tmask = cv2.imread(os.path.join(path, 'steps','mask_step_025.png'))
    tmask[tmask==tmask.max()] = 0
    tmask[tmask>0] = 255
    tmask[mask==0] = 0
    edge = cv2.dilate(edge, kernel=np.ones((3,3))).sum(-1).clip(0,255)[...,None].repeat(3,2)
    
    im = preprocess(img)
    mk = preprocess(mask)
    ed = preprocess(edge)
    tk = preprocess(tmask)
    
    cond = mk*ed*0.1 + (1-mk)*im + (1-ed)*tk
    return cond

def test():
    model = ControlModel(256,512,8).cuda()
    model.load_state_dict(torch.load('/mnt/share/shenfeihong/weight/smile-sim/2023.5.26/080000.pt')["g_ema"])
    
    cond = get_cond_x().unsqueeze(0).cuda()
    noise = [torch.randn(1,512).cuda()]
    img, _ = model(cond, noise)
    utils.save_image(
        img,
        f"img.jpg",
        nrow=1,
        normalize=True,
        range=(0, 1),
    )
    utils.save_image(
        cond,
        f"cond.jpg",
        nrow=1,
        normalize=True,
        range=(0, 1),
    )
if __name__=='__main__':
    # path = '/mnt/share/shenfeihong/data/smile_to_b_test/x'
    # img = cv2.imread(os.path.join(path, 'mouth.jpg'))
    # mask = cv2.imread(os.path.join(path, 'MouthMask.png'))
    # edge = cv2.imread(os.path.join(path, 'steps', 'step_025.png'))
    # tmask = cv2.imread(os.path.join(path, 'steps', 'mask_step_025.png'))
    # tmask[tmask==tmask.max()] = 0
    # tmask[tmask>0] = 255
    # # tmask[mask==0] = 0
    # cv2.imwrite('img.jpg', tmask)
    test()