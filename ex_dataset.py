from torch.utils.data import Dataset
# from PIL import Image
# from utils import data_utils
import os
import cv2
import numpy as np
import random
import math

class RandomPerspective:

    def __init__(self,
                 degrees=0.0,
                 translate=0.0,
                 scale=0.0,
                 shear=0.0,
                 perspective=1.0,
                 border=(0, 0),
                 pre_transform=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        # Mosaic border
        self.border = border
        self.pre_transform = pre_transform
        self.size = (256,256)
    def __call__(self, img):
        """Center."""
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(-self.translate, self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(-self.translate, self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        # M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        M = T @ S @ R
        # print(M)
        
        # Affine image

        if self.perspective:
            img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(0, 0, 0))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(0, 0, 0))
        return img

class ImagesDataset(Dataset):
    def __init__(self, mode='train'):
        self.path = '/ssd/gregory/smile/YangNew/'
        # self.path = '/mnt/d/data/smile/out'
        self.all_files = []
        if mode=='train':
            for folder in os.listdir(self.path)[5:]:
                self.all_files.append(os.path.join(self.path, folder,))
        else:
            for folder in os.listdir(self.path)[:5]:
                self.all_files.append(os.path.join(self.path, folder,))
            
            # if os.path.exists(os.path.join(self.path, folder, 'modal', 'blend.png')):
                # self.all_files.append(os.path.join(self.path, folder, 'modal'))
        print('total image:', len(self.all_files))
        self.mode = mode
        self.half = False
        self.trans = RandomPerspective(degrees=0.0, translate=0, scale=0.0)
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        img_folder = self.all_files[index]
        img = cv2.imread(os.path.join(img_folder, 'Img.jpg'))
        mask = cv2.imread(os.path.join(img_folder, 'MouthMask.png'))
        # teeth_3d = cv2.imread(os.path.join(img_folder, 'teeth_3d.png'))
        edge = cv2.imread(os.path.join(img_folder, 'TeethEdge.png'))
        
        
        im = self.preprocess(img)
        mk = self.preprocess(mask)
        # teeth_3d = self.preprocess(teeth_3d)
        ed = self.preprocess(edge)
        
        input = np.zeros((5,256,256))
        cond = self.trans(img)
        cond[mask==0]=0
        cond = self.preprocess(cond)
        input[0]=mk[0]
        input[1]=ed[0]
        input[-3:]=im
        return {'images': im, 'input':input, 'cond':cond}
        
    def preprocess(self, img):
        img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        if len(img_resize.shape)==2:
            img_resize = img_resize[:,:,None].repeat(1,1,3)
        im = np.ascontiguousarray(img_resize.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = im / 255.0  # 0-255 to 0.0-1.0
        return im

class TestDataset(Dataset):
    def __init__(self, mode='train'):
        self.path = '/mnt/d/data/smile/out'
        # self.path = '/mnt/d/data/smile/out'
        self.all_files = []

        for folder in os.listdir(self.path):
            self.all_files.append(os.path.join(self.path, folder,))
            
            # if os.path.exists(os.path.join(self.path, folder, 'modal', 'blend.png')):
                # self.all_files.append(os.path.join(self.path, folder, 'modal'))
        print('total image:', len(self.all_files))
        self.mode = mode
        self.half = False
        # self.trans = RandomPerspective(degrees=0.0, translate=0, scale=0.0)
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        img_folder = self.all_files[index]
        img = cv2.imread(os.path.join(img_folder, 'smile.png'))
        mask = cv2.imread(os.path.join(img_folder, 'mouth_mask.png'))
        edge = cv2.imread(os.path.join(img_folder, 'edge.png'))
        
        
        im = self.preprocess(img)
        mk = self.preprocess(mask)
        # teeth_3d = self.preprocess(teeth_3d)
        ed = self.preprocess(edge)
        
        input = np.zeros((5,256,256))
        input[0]=mk[0]
        input[1]=ed[0]
        input[-3:]=im
        return {'images': im, 'input':input}
        
    def preprocess(self, img):
        img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        if len(img_resize.shape)==2:
            img_resize = img_resize[:,:,None].repeat(1,1,3)
        im = np.ascontiguousarray(img_resize.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = im / 255.0  # 0-255 to 0.0-1.0
        return im
import torch
class MyTestDataset(Dataset):
    def __init__(self, mode='test'):
        self.path = '/mnt/d/data/smile/out'
        # self.path = '/ssd/gregory/smile/YangNew/'
        
        self.all_files = []
        folder_list = os.listdir(self.path)
            
        for folder in folder_list:
            self.all_files.append(os.path.join(self.path, folder,))
            
            # if os.path.exists(os.path.join(self.path, folder, 'modal', 'blend.png')):
                # self.all_files.append(os.path.join(self.path, folder, 'modal'))
        print('total image:', len(self.all_files))
        self.mode = mode
        self.half = False
        self.aug = RandomPerspective(translate=0.05, degrees=5, scale=0.05)
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        img_folder = self.all_files[index]
        img = cv2.imread(os.path.join(img_folder, 'smile.png'))
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        im = self.preprocess(img)
        cond_im = img.copy()
        
        
        cond = np.zeros((7,256,256))
        ed = cv2.imread(os.path.join(img_folder, 'down_edge.png'))
        cond[0] = self.preprocess(ed)[0]
        eu = cv2.imread(os.path.join(img_folder, 'upper_edge.png'))
        cond[1] = self.preprocess(eu)[0]
        mk = cv2.imread(os.path.join(img_folder, 'mouth_mask.png'))
        cond[2] = self.preprocess(mk)[0]
        tk = cv2.imread(os.path.join(img_folder, 'teeth_mask.png'))
        cond[3] = self.preprocess(tk)[0]
        
        cond_im[tk==0]=0
        cond_im = self.aug(cond_im)
        # cv2.imshow('img',cond_im)
        # cv2.waitKey(0)
        cond[-3:] = self.preprocess(cond_im)        
        return {'images': im, 'cond':cond}
    def preprocess(self, img):
        img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        if len(img_resize.shape)==2:
            img_resize = img_resize[:,:,None].repeat(1,1,3)
        im = np.ascontiguousarray(img_resize.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
if __name__=='__main__':
    ds = ImagesDataset()
    for batch in ds:
        break