# -*-coding: utf-8 -*-

import os, sys
import os.path as osp

import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import time
import math
from PIL import Image
from config import base_config
from visualize import save_pred,get_palette
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def vis_res(image, preds):
    #res = np.squeeze(res)
    palette = get_palette(256)
    res_num = preds.shape[1]
    new_imge = Image.new('RGBA', (preds.shape[2] * (preds.shape[1]+1), preds.shape[3] * 1))
    new_imge.paste(Image.fromarray(((np.transpose(np.array(image[0, ...]), (1, 2, 0)))).astype(np.uint8)), (0, 0))
    for i in range(0, res_num):
        mask_i = Image.fromarray((preds[0, i, ...]).astype(np.uint8))
        mask_i.putpalette(palette)
        new_imge.paste(mask_i, (mask_i.size[0] * (i+1), 0))
    #teeth_up.putpalette(palette)
    #teeth_down = Image.fromarray((preds[0, 2, ...]).astype(np.uint8))
    #teeth_down.putpalette(palette)
    #teeth_updown = Image.fromarray( np.array(teeth_up) + 2*np.array(teeth_down))
    #teeth_updown.putpalette(palette)


    #new_imge.paste(teeth_down, (teeth_down.size[0] * (2), 0))
    plt.imshow(new_imge)
    plt.show()

class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.onnx_session.set_providers(['CUDAExecutionProvider'],[{'device_id':0}])
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores
CLASSES = ['MouthMask', 'TeethEdge', 'TeethEdgeDown', 'TeethEdgeUp', 'TeethMasks']
def get_palette( n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette
label_map = {0:"MouthMask", 1:"TeethEdge", 2:"TeethEdgeDown", 3:"TeethEdgeUp", 4:"TeethMasks"}


def to_numpy(tensor):
    print(tensor.device)
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


#img_dir = r"G:\data_zzz\data_tooth"
img_txt = base_config.test_path
with open(img_txt, 'r') as f:
    image_lines = f.readlines()
images_path = list(map(lambda x: x.split(' ')[0], image_lines))
timestamp = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
img_size = (256,256)
r_model_path = r"onnx/tooth_model_2021-11-04_15_55_44.onnx"
save_path = r"onnx\res_tooth_model_{}".format(timestamp)
os.makedirs(save_path, exist_ok=True)
time_start1 = time.time()
rnet1 = ONNXModel(r_model_path)
time_end2 = time.time()
print('load model cost', time_end2 - time_start1)

# 测时间
for img_path in images_path:
    masks = []
    for mask_name in CLASSES:
        masks.append(cv2.imread(osp.join(osp.dirname(img_path), mask_name + '.png'), 0))
    masks = np.array(masks)

    masks = masks[np.newaxis,...]
    time_start = time.time()
    img_ori = cv2.imread(img_path)
    # if(img_ori == None):
    #     continue
    try:
        img = cv2.resize(img_ori, img_size)
    except:
        print(img_path, "error")
        continue
    img = img[..., ::-1] # BGR to RGB
    img_input = (np.float32(img)/255.0)
    #img_input = (img_input - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_input = img_input.transpose((2, 0, 1))
    img_input = torch.from_numpy(img_input).unsqueeze(0)
    img_input = img_input.type(torch.FloatTensor)

    out = rnet1.forward(to_numpy(img_input))[0]
    out = out > 0.4
    print("output shape:", out.shape)

    time_end=time.time()
    print('infer cost',time_end-time_start)


    name = osp.basename(osp.dirname(img_path))
    #save_pred((np.transpose(img, (2,0,1))[np.newaxis,...]), out, masks, save_path, name, tmp_iou=None)
    vis_res((np.transpose(img, (2,0,1))[np.newaxis,...]), out)