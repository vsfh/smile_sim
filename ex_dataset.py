from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import os
import cv2
import numpy as np
class ImagesDataset(Dataset):
    def __init__(self, mode='train'):
        self.path = '/mnt/d/data/smile/out'
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
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        img_folder = self.all_files[index]
        img = cv2.imread(os.path.join(img_folder, 'mouth.png'))
        mask = cv2.imread(os.path.join(img_folder, 'mouth_mask.png'))
        teeth_3d = cv2.imread(os.path.join(img_folder, 'teeth_3d.png'))
        
        im = self.preprocess(img)
        mk = self.preprocess(mask)
        teeth_3d = self.preprocess(teeth_3d)
        
        cond = teeth_3d*mk+im*(1-mk)
        
        return {'images': im, 'cond':cond}
        
    def preprocess(self, img):
        img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        if len(img_resize.shape)==2:
            img_resize = img_resize[:,:,None].repeat(1,1,3)
        im = np.ascontiguousarray(img_resize.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = im / 255.0  # 0-255 to 0.0-1.0
        return im