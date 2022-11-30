import os

import numpy
import numpy as np
from PIL import Image
import cv2
import torch
import onnxruntime


def label_check():
    base_path = '/media/vsfh/7EF4D15FF4D11A6B/sfh/dataset/teeth_img/filtered_smile_img'
    mask_path = '/media/vsfh/7EF4D15FF4D11A6B/sfh/dataset/teeth_img/filtered_smile_mask'
    name_list = os.listdir(base_path)
    for name in name_list:
        file_path = os.path.join(base_path, 'C01002721732_smile.jpg')
        img = cv2.imread(file_path)
        img = cv2.resize(img, (256, 256))
        label_path = os.path.join(mask_path, 'C01002721732_smile.jpg')
        label = cv2.imread(label_path).astype(bool)
        img[label] = 255
        cv2.imshow('check', img)
        cv2.waitKey(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
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


def segmentation(img_dir):
    name_list = []
    r_model_path = "/home/vsfh/training/tooth_model_2021-11-04_15_55_44.onnx"
    rnet1 = ONNXModel(r_model_path)
    base_path = '/home/vsfh/dataset/a_down'
    file_list = os.listdir(base_path)
    for idx in range(1):
        file = file_list[idx]
        file_path = os.path.join(base_path, file)
        part_face = cv2.imread('/home/vsfh/dataset/smile/C01002721732_smile.jpg')
        part_face = part_face[..., ::-1]
        img_input = (np.float32(part_face) / 255.0)
        img_input = img_input.transpose((2, 0, 1))
        img_input = torch.from_numpy(img_input).unsqueeze(0)

        img_input = img_input.type(torch.FloatTensor)
        out = rnet1.forward(to_numpy(img_input))[0]
        for i in range(out.shape[1]):
            pred = out[0, i, ...]
            pic = (pred > 0.4).astype(np.uint8)
            cv2.imwrite(f'./seg_test/{i}.jpg', pic * 255)


def detec():
    name_list = []
    r_model_path = "/home/vsfh/training/tooth_model_2021-11-04_15_55_44.onnx"
    rnet1 = ONNXModel(r_model_path)
    base_path = '/home/vsfh/dataset/a_down'
    file_list = os.listdir(base_path)
    for file in file_list:
        # file = file_list[idx]
        file_path = os.path.join(base_path, file)
        part_face = cv2.imread(file_path)
        part_face = part_face[..., ::-1]
        img_input = (np.float32(part_face) / 255.0)
        img_input = img_input.transpose((2, 0, 1))
        img_input = torch.from_numpy(img_input).unsqueeze(0)

        img_input = img_input.type(torch.FloatTensor)
        out = rnet1.forward(to_numpy(img_input))[0]
        pred = out[0, 4, ...]
        pic = (pred > 0.4).astype(np.uint8)
        a = []
        for i in range(255):
            if (pic[:, i] == False).all() != (pic[:, i + 1] == False).all():
                a.append(i)
        inner = out[0, 0, ...]
        inner_pic = (inner > 0.4)
        inner_pic[:, a[0]:a[-1]] = 0
        part_face = part_face[..., ::-1]
        part_face[inner_pic, 1] = 127
        # part_face[inner_pic,:] = part_face[inner_pic,:] + 1000
        cv2.imwrite(f'./seg_test/detec.jpg', part_face)


def upper_gums():
    name_list = []
    r_model_path = "/home/vsfh/training/tooth_model_2021-11-04_15_55_44.onnx"
    rnet1 = ONNXModel(r_model_path)
    base_path = '/home/vsfh/dataset/a_down'
    file_list = os.listdir(base_path)
    for idx in range(1):
        file = file_list[idx]
        file_path = os.path.join(base_path, file)
        part_face = cv2.imread('/home/vsfh/dataset/smile/C01002721675_smile.jpg')
        part_face = part_face[..., ::-1]
        img_input = (np.float32(part_face) / 255.0)
        img_input = img_input.transpose((2, 0, 1))
        img_input = torch.from_numpy(img_input).unsqueeze(0)

        img_input = img_input.type(torch.FloatTensor)
        out = rnet1.forward(to_numpy(img_input))[0]
        pred = out[0, 4, ...]
        inner = out[0, 0, ...]
        inner_pic = (inner > 0.4).astype(np.uint8)
        pic = (pred > 0.4).astype(np.uint8)
        a = []
        upper_gum = 0
        lower_gum = 0
        for i in range(255):
            if (pic[:, i] == False).all() != (pic[:, i + 1] == False).all():
                a.append(i)
        part_face = part_face[..., ::-1]
        for i in range(a[0], a[-1]):
            inner_ = np.where(pic[:, i+1] == True)
            teeth_ = np.where(inner_pic[:, i+1] == True)
            # pic[inner_[0][0], i+1] = 120
            # pic[inner_[0][-1], i+1] = 120
            #
            # inner_pic[teeth_[0][0], i] = 120
            # inner_pic[teeth_[0][-1], i] = 120
            part_face[teeth_[0][0]:inner_[0][0], i+1, 0] = 120
            part_face[inner_[0][-1]:teeth_[0][-1], i+1, 2] = 120
            upper_gum += inner_[0][0] - teeth_[0][0]
            lower_gum += teeth_[0][-1] - inner_[0][-1]

        cv2.imshow('inner', part_face)
        cv2.waitKey(0)

def batch_seg(path):
    from test import Yolo, Segmentation, sigmoid
    from utils import loose_bbox

    seg = Segmentation('/mnt/share/shenfeihong/weight/pretrain/edge.onnx', (256, 256))
    img_dir = os.path.join(path,'smile_6000')
    for file in os.listdir(img_dir)[:2000]:
        img_path = os.path.join(img_dir,file)
        image = cv2.imread(img_path)
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = seg.predict(image)

        mask = (result[..., 1] > 0.6).astype(np.uint8)*255
        img_name = img_path.split('/')[-1].split('_')[0]
        print(img_name)
        os.makedirs(os.path.join(path,'seg_6000',img_name), exist_ok=True)
        # cv2.imwrite(os.path.join(path,'seg_6000',img_name,'mouth.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join(path,'seg_6000',img_name,'MouthMask.png'), mask)
        cv2.imwrite(os.path.join(path,'seg_6000',img_name,'edge.png'), mask)
        
def mask_filter(path):
    import cv2
    import numpy as np
    import skimage.exposure

    for img_name in os.listdir(os.path.join(path,'seg_6000')):
        print(img_name)
        img = cv2.imread(os.path.join(path,'seg_6000',img_name,'MouthMask.png'))

        # blur threshold image
        blur = cv2.GaussianBlur(img, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)

        result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))

        # save output
        cv2.imwrite(os.path.join(path,'seg_6000',img_name,'mask_filtered.png'), result)
        # image = Image.open()
        # image = image.filter(ImageFilter.ModeFilter(size=13))
        # image.save()
    
if __name__ == '__main__':
    # segmentation('a')
    # upper_gums()
    # label_check()
    batch_seg('/mnt/share/shenfeihong/data/smile_')
