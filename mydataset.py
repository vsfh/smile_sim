from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import torch

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class YangOldNew(Dataset):
    def __init__(self, mode='decoder'):
        self.path = '/data/shenfeihong/smile/Merged/Teeth'
        self.all_files = os.listdir(self.path)
        print('total image:', len(self.all_files))
        self.mode = mode
        self.half = False
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        img_folder = self.all_files[index]
        img = cv2.imread(os.path.join(self.path, img_folder, 'Img.jpg'))
        mask = cv2.imread(os.path.join(self.path, img_folder, 'MouthMask.png'))
        edge = cv2.imread(os.path.join(self.path, img_folder, 'TeethEdge.png'))
        tmask = cv2.imread(os.path.join(self.path, img_folder, 'TeethMasks.png'))
        

        im = self.preprocess(img)
        mk = self.preprocess(mask)
        ed = self.preprocess(edge)
        tk = self.preprocess(tmask)
        
        
        return {'images': im, 'mask': mk, 'edge':ed, 'tmask':tk}
        
    def preprocess(self, img):
        img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        if len(img_resize.shape)==2:
            img_resize = img_resize[:,:,None].repeat(1,1,3)
        im = np.ascontiguousarray(img_resize.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
        
class Tianshi(Dataset):
    def __init__(self, mode='decoder'):
        self.path_1 = '/data/shenfeihong/tianshi_seg/'
        self.path_2 = '/data/shenfeihong/tianshi_1.4_seg/'
        self.all_files = os.listdir(self.path_1)+os.listdir(self.path_2)
        print('total image:', len(self.all_files))
        self.mode = mode
        self.half = False
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        img_folder = self.all_files[index]
        img = cv2.imread(os.path.join(self.path, img_folder, 'mouth.jpg'))
        mask = cv2.imread(os.path.join(self.path, img_folder, 'mask.png'))
        edge = cv2.imread(os.path.join(self.path, img_folder, 'edge.png'))
        up_edge = cv2.imread(os.path.join(self.path, img_folder, 'up_edge.png'))
        
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmask = np.zeros_like(edge)
        cv2.drawContours(tmask, contours, -1, (255), thickness=cv2.FILLED)

        im = self.preprocess(img)
        mk = self.preprocess(mask)
        ed = self.preprocess(edge)
        tk = self.preprocess(tmask)
        up = self.preprocess(up)
        
        cond = 0.1*ed*mk+0.5*up*mk+(1-mk)*im+(1-ed)*tk
        return {'images': im, 'cond': cond}
        
        # return {'images': im, 'mask': mk, 'edge':ed, 'tmask':tk}
        
    def preprocess(self, img):
        img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        if len(img_resize.shape)==2:
            img_resize = img_resize[:,:,None].repeat(1,1,3)
        im = np.ascontiguousarray(img_resize.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
    
def mask_proc(mask, dilate=True):
        mask = np.array(mask)/255
        if len(mask.shape) == 3:
            mask = mask[...,0]

        return mask.astype(np.float32)[None]

def seg_proc(tid_seg, show_tid):
    # show_tid = [11,21,41,31]
    out_tid = np.zeros_like(tid_seg)
    for tid in show_tid:
        out_tid[tid_seg==tid]=1
    # tid_seg[tid_seg!=1]=0
    return out_tid.astype(np.float32)[None]


class edge(Dataset):
    def __init__(self, mode='train'):
        
        if mode=='train':
            self.all_files = natsorted(glob.glob('/mnt/share/shenfeihong/data/smile_/2022-11-30-cvat/face_seg_22_11_25/*/mouth.png', recursive=False))[40:]
        elif mode=='val':
            self.all_files = natsorted(glob.glob('/mnt/share/shenfeihong/data/smile_/2022-11-30-cvat/face_seg_22_11_25/*/mouth.png', recursive=False))[:40]
        else:
            self.all_files = natsorted(glob.glob('/mnt/share/shenfeihong/data/smile_/2022-11-30-cvat/dataset_test/*/mouth.png', recursive=False))[:40]
            
        print('total image:', len(self.all_files))
        self.transform = transforms.Compose(
                    [
                        transforms.Resize([256,256]),
                        transforms.ToTensor()
                        
                    ]
            )

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        ori_file = self.all_files[index]
        
        mask_file = ori_file.replace('mouth', 'mask')
        tid_file = ori_file.replace('mouth', 'tid')
        
        up_edge = ori_file.replace('mouth', 'up_edge')
        down_edge = ori_file.replace('mouth', 'down_edge')
        
        up_edge = mask_proc(cv2.imread(up_edge))
        down_edge = mask_proc(cv2.imread(down_edge))
        
        mask = mask_proc(cv2.imread(mask_file))
        tid_seg = np.array(cv2.imread(tid_file))[...,0]

        label = np.zeros((256,256))
        label[tid_seg==11]=1.0
        label[tid_seg==21]=1.0
        label = label.astype(np.float32)[None]
        
        img = Image.open(ori_file)
        img = self.transform(img)*2-1
            
        if np.random.randint(2)>0:
            mask = mask[:,:,::-1].copy()
            img = flip(img,2)
            label = label[:,:,::-1].copy()
            up_edge = up_edge[:,:,::-1].copy()
            down_edge = down_edge[:,:,::-1].copy()
            

        return {'mask': mask, 'mouth':img, 'label':label, 'up_edge':up_edge+down_edge}

class pair(Dataset):
    def __init__(self, mode='train'):
        

        self.all_files = (glob.glob('/data/shenfeihong/pair/*/mouth.png', recursive=False))

            
        print('total image:', len(self.all_files))
        self.transform = transforms.Compose(
                    [
                        transforms.Resize([256,256]),
                        transforms.ToTensor()
                        
                    ]
            )

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        ori_file = self.all_files[index]
        
        mask_file = ori_file.replace('mouth', 'mask')
        
        align_file = ori_file.replace('mouth', 'align')
        
        mask = mask_proc(cv2.imread(mask_file))

        img = Image.open(ori_file)
        img = self.transform(img)*2-1

        align_img = Image.open(align_file)
        align_img = self.transform(align_img)*2-1
         
        if np.random.randint(2)>0:
            mask = mask[:,:,::-1].copy()
            img = flip(img,2)
            align_img = flip(align_img,2)
            
        return {'mask': mask, 'mouth':img, 'align_img':align_img}
        
def get_loader_unet(size =1, mode='train'):
    dataset = edge(mode)
    loader = DataLoader(
        dataset=dataset,
        batch_size=size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    return loader
    
if __name__ == '__main__':
    # ds = SmileDataset(r'C:\data\smile_synthesis\smile_segmap', None)
    # ds = SimulationDataset(r'C:\data\smile_synthesis\smile_segmap', None)
    ds = YangOldNew()

    for batch in ds:
        images = batch['images']
        mask = batch['mask']
        edge = batch['edge']
        print(edge.shape)
        a = np.zeros((256,256,3))
        a[...,0] = mask*255
        a[...,1] = edge*255
        # print(images.shape, mask.shape)
        assert images.shape[1]==images.shape[2], 'error'
        assert mask.shape[1]==mask.shape[2], 'mask error'

        # print(set(input_semantic.flatten()))
        # print(set(batch['mask'].flatten()))

        # print(input_semantic.shape)
        # for i, o in enumerate(input_semantic):
        #     cv2.imshow(str(i), o)
        #     print(i, o.min(), o.max())
        # # cv2.imshow('tmp', input_semantic[:3].transpose(1,2,0))
        cv2.imshow('img', a.astype(np.uint8))
        cv2.waitKey(0)


