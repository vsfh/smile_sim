from control_gan import ControlModel
import torch
import cv2
import numpy as np
from PIL import Image
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
    
def apply_style(img, mean, std, mask=None, depth=None):
    if len(mask.shape) == 2:
        mask = mask[..., None]
    if mask.shape[2] == 3:
        mask = mask[..., :1]
    mask = mask.astype(np.uint8)

    # img_depth = img.astype(np.float32) * (depth[..., None] * 0.6 + 0.4)
    # img_depth = img_depth.astype(np.uint8)

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype("float32")

    src_mean, src_std = cv2.meanStdDev(img_lab, mask=mask)
    img_lab = (img_lab - src_mean.squeeze()) / src_std.squeeze() * std.squeeze() + mean.squeeze()
    img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    img_rgb = img_rgb.astype(np.float32)
    # * (depth[...,None] * 0.3 + 0.7)* (depth[...,None] * 0.3 + 0.7)

    mask = mask.astype(np.float32)

    smooth_mask = cv2.blur(mask, (5, 5))
    smooth_mask = smooth_mask[..., None]

    result = img_rgb * smooth_mask + (1 - smooth_mask) * img
    result = result.astype(np.uint8)

    return result

def get_cond(path):
    #  = '/mnt/e/data/smile/to_b/test_63/test_03_26/C01004891833'
    img = cv2.imread(os.path.join(path, 'mouth.jpg'))
    mask = cv2.imread(os.path.join(path, 'MouthMask.png'))
    edge = cv2.imread(os.path.join(path, 'steps', 'step_036.png'))
    # tmask = cv2.imread(os.path.join(path, 'steps', 'depth_058.png'))[...,0]
    # edge = edge.sum(-1).repeat(3,2)
    # tmask[tmask==tmask.max()] = 0
    # tmask[tmask>0] = 255
    # tmask[mask[...,0]==0] = 0
    up_edge = edge[...,2]
    down_edge = edge[...,1]
    
    edge = edge.sum(-1).clip(0,255).astype(np.uint8)
    
    contours1, _ = cv2.findContours(up_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(down_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tmask = np.zeros_like(edge)
    cv2.drawContours(tmask, contours1, -1, (1), thickness=cv2.FILLED)
    cv2.drawContours(tmask, contours2, -1, (1), thickness=cv2.FILLED)
    
    # edge = cv2.dilate(edge, kernel=np.ones((3,3)))
    # up_edge = cv2.dilate(up_edge, kernel=np.ones((3,3)))
    # edge = cv2.dilate(edge, kernel=np.ones((3,3)))
    
    edge = edge[...,None].repeat(3,2)
    up_edge = up_edge[...,None].repeat(3,2)
    down_edge = down_edge[...,None].repeat(3,2)
    
    tmask = tmask[...,None].repeat(3,2)
    
    im = preprocess(img)
    mk = preprocess(mask)
    up = preprocess(up_edge)
    down = preprocess(down_edge)
    ed = preprocess(edge)
    tk = preprocess(tmask)
    
    # cond = mk*down*0.1 + mk*up*0.5 + (1-mk)*im + (1-ed)*tk
    cond = mk*ed*0.1 + mk*up*0.5 + (1-mk)*im + (1-ed)*mk*tk
    utils.save_image(
        cond,
        f"cond.jpg",
        nrow=1,
        normalize=True,
        range=(0, 1),
    )
    


    return cond, mk
    
def get_cond_x():
    path = '/mnt/share/shenfeihong/data/smile_to_b_test/x'
    path = '/mnt/share/shenfeihong/data/TrainDataNew/Teeth/C01003324642'
    img = cv2.imread(os.path.join(path, 'Img.jpg'))
    mask = cv2.imread(os.path.join(path, 'MouthMask.png'))
    edge = cv2.imread(os.path.join(path, 'TeethEdge.png'))
    tmask = cv2.imread(os.path.join(path, 'TeethMasks.png'))

    
    im = preprocess(img)
    mk = preprocess(mask)
    ed = preprocess(edge)
    tk = preprocess(tmask)
    
    cond = mk*ed*0.1 + (1-mk)*im + (1-ed)*tk
    return cond, mk

def test():
    model = ControlModel(256,512,8,export=True).cuda()
    model.load_state('/mnt/share/shenfeihong/weight/smile-sim/2023.6.13/030000.pt')
    
    path = '/mnt/e/data/smile/to_b/test_63/test_03_26'
    for case in os.listdir(path):
        img_folder = os.path.join(path,case)
        cond, mk = get_cond(img_folder)
        cond = cond.unsqueeze(0).cuda()
        mk = mk.unsqueeze(0).cuda()
        noise = [torch.randn(1,512).cuda()]
        img = model(cond)
        utils.save_image(
            img,
            os.path.join(path,case,'sfhRes/smile.jpg'),
            nrow=1,
            normalize=True,
            range=(0, 1),
        )
        print(case)
    # utils.save_image(
    #     cond,
    #     f"cond.jpg",
    #     nrow=1,
    #     normalize=True,
    #     range=(0, 1),
    # )
def test_onnx():
    import onnxruntime
    model = onnxruntime.InferenceSession('model2.onnx',
                                    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                               'CPUExecutionProvider'])
    path = '/mnt/e/data/smile/to_b/test_63/test_03_26'
    for case in os.listdir(path):
        case = 'C01004899864'
        img_folder = os.path.join(path,case)
        cond, mk = get_cond(img_folder)
        cond = cond.unsqueeze(0).numpy()
        mk = mk.unsqueeze(0).numpy()
        noise = [torch.randn(1,512).cuda()]
        img = model.run([], {
                'cond': cond
            })
        sample = img[0][0]
        sample = cond[0]*(1-mk[0]) + sample*mk[0]
        sample = np.clip(sample, 0, 1)
        sample = sample.transpose(1, 2, 0)
        sample = (sample * 255).astype(np.uint8)
        
        tmask_ori = cv2.imread(os.path.join(img_folder, 'TeethMasks.png'))
        tmask_ori = tmask_ori[..., 0]
        img = np.array(Image.open(os.path.join(img_folder, 'mouth.jpg')))
        mouth_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        target_mean, target_std = cv2.meanStdDev(mouth_lab, mask=tmask_ori)
        depth = np.array(Image.open(os.path.join(img_folder, 'steps', 'depth_036.png')).convert('L'))
        depth = depth.astype(np.float32) / 255
        
        apply_mask = depth.copy()
        apply_mask[depth>0] = 1
        apply_mask = apply_mask
        sample = apply_style(sample, target_mean, target_std, apply_mask*mk[0][0], depth)
        cv2.imwrite(os.path.join(img_folder,'paper_ex','smile2.jpg'), sample[...,::-1])
        break
def onnx_export():
    dynamic_axes = {
        'cond': {0: 'batch_size'},
        'align_img': {0: 'batch_size'}
    }
    model = ControlModel(256,512,8,export=True).eval().cuda()
    model.load_state('/mnt/e/share/035000.pt')

    input1 = torch.randn(1,3,256,256).cuda()
    input_name = ['cond']
    # input_name = ['input_image','mask','big_mask']
    
    output_name = ['align_img']
    torch.onnx.export(model, (input1), 'model2.onnx', export_params=True, input_names=input_name, output_names=output_name,
                      opset_version=16, dynamic_axes=dynamic_axes)
if __name__=='__main__':

    test_onnx()
