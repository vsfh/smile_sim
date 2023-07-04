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
    if im.max()>50:
        im /= 255.0  # 0-255 to 0.0-1.0
    return im
    
def get_cond():
    path = '/mnt/share/shenfeihong/data/smile_to_b_test/x'
    path = '/mnt/share/shenfeihong/data/smile_to_b_test/test_03_26/C01004898447'
    img = cv2.imread(os.path.join(path, 'mouth.jpg'))
    mask = cv2.imread(os.path.join(path, 'MouthMask.png'))
    edge = cv2.imread(os.path.join(path, 'steps', 'step_065.png'))
    # tmask = cv2.imread(os.path.join(path, 'steps', 'depth_065.png'))
    # edge = edge.sum(-1).repeat(3,2)
    # tmask[tmask==tmask.max()] = 0
    # tmask[tmask>0] = 255
    # tmask[mask==0] = 0
    cv2.imshow('edge0', edge[...,0])
    cv2.imshow('edge0', edge[...,0])
    cv2.imshow('edge0', edge[...,0])
    cv2.waitKey(0)
    edge = cv2.dilate(edge, kernel=np.ones((3,3))).sum(-1).clip(0,255).astype(np.uint8)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask to fill the white edge
    tmask = np.zeros_like(edge)

    # Draw filled contours on the mask
    cv2.drawContours(tmask, contours, -1, (255), thickness=cv2.FILLED)
    edge = edge[...,None].repeat(3,2)
    tmask = tmask[...,None].repeat(3,2)
    
    mask = mask/255
    edge = edge/255
    tmask = tmask/255
    img = img/255
    cond = mask*edge*0.1 + (1-mask)*img + (1-edge)*tmask
    cv2.imshow('img', cond)
    cv2.waitKey(0)
    
    im = preprocess(img)
    mk = preprocess(mask)
    ed = preprocess(edge)
    tk = preprocess(tmask)
    
    cond = mk*ed*0.1 + (1-mk)*im + (1-ed)*tk

    return cond, mk
    
def get_cond_x():
    path = '/mnt/share/shenfeihong/data/smile_to_b_test/x'
    path = '/mnt/share/shenfeihong/data/TrainDataNew/Teeth/C01003324642'
    img = cv2.imread(os.path.join(path, 'Img.jpg'))
    mask = cv2.imread(os.path.join(path, 'MouthMask.png'))
    edge = cv2.imread(os.path.join(path, 'TeethEdge.png'))
    tmask = cv2.imread(os.path.join(path, 'TeethMasks.png'))
    
    # tmask[tmask==tmask.max()] = 0
    # tmask[tmask>0] = 255
    # tmask[mask==0] = 0
    # edge = cv2.dilate(edge, kernel=np.ones((3,3))).sum(-1).clip(0,255)[...,None].repeat(3,2)
    
    # mask = mask/255
    # edge = edge/255
    # tmask = tmask/255
    # img = img/255
    # cond = mask*edge*0.1 + (1-mask)*img + (1-edge)*tmask
    # cv2.imshow('img', cond)
    # cv2.waitKey(0)
    
    im = preprocess(img)
    mk = preprocess(mask)
    ed = preprocess(edge)
    tk = preprocess(tmask)
    
    cond = mk*ed*0.1 + (1-mk)*im + (1-ed)*tk
    return cond, mk

def test():
    model = ControlModel(256,512,8,export=True).cuda()
    model.load_state('/mnt/share/shenfeihong/weight/smile-sim/2023.6.25/040000.pt')
    
    cond, mk = get_cond_x()
    cond = cond.unsqueeze(0).cuda()
    mk = mk.unsqueeze(0).cuda()
    noise = [torch.randn(1,512).cuda()]
    img = model(cond)
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

def onnx_export():
    dynamic_axes = {
        'cond': {0: 'batch_size'},
        'align_img': {0: 'batch_size'}
    }
    model = ControlModel(256,512,8,export=True).eval().cuda()
    model.load_state('/mnt/share/shenfeihong/weight/smile-sim/2023.5.26/080000.pt')

    input1 = torch.randn(1,3,256,256).cuda()
    input_name = ['cond']
    # input_name = ['input_image','mask','big_mask']
    
    output_name = ['align_img']
    torch.onnx.export(model, (input1), 'model.onnx', export_params=True, input_names=input_name, output_names=output_name,
                      opset_version=16, dynamic_axes=dynamic_axes)
if __name__=='__main__':
    # path = '/mnt/share/shenfeihong/data/smile_to_b_test/x'
    # ed = cv2.imread('example/1.4.tianshi/598236/edge.png')
    # down = cv2.imread('example/1.4.tianshi/598236/down_edge.png')
    # up = cv2.imread('example/1.4.tianshi/598236/up_edge.png')
    
    # cv2.imshow('img', ed-up)
    # cv2.waitKey(0)
    # edge = cv2.imread(os.path.join(path, 'steps', 'step_025.png'))
    # tmask = cv2.imread(os.path.join(path, 'steps', 'mask_step_025.png'))
    # tmask[tmask==tmask.max()] = 0
    # tmask[tmask>0] = 255
    # # tmask[mask==0] = 0
    # cv2.imwrite('img.jpg', tmask)
    # a,b = get_cond()
    test()