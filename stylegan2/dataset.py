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
class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
        
class img_dataset(Dataset):
    def __init__(self):
        root_dir = r'D:\sfh\dataset\teeth_img'
        self.img_dir = os.path.join(root_dir, 'filtered_smile_img')
        # flip_dir = os.path.join(root_dir, 'flip_filtered_smile_img')
        name_list = []
        for img_name in os.listdir(self.img_dir):
            name_list.append(os.path.join(self.img_dir, img_name))
            # name_list.append(os.path.join(flip_dir, 'flip_'+img_name))
        self.list_ids = name_list

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        img_name = self.list_ids[index]
        img = Image.open(img_name)
        img = np.array(img).astype(np.float32) / 255 * 2 - 1

        return {
            'img': img.transpose(2, 0, 1),
        }

class no_conditional(Dataset):
    def __init__(self):
        root_dir = r'D:\sfh\dataset\teeth_img'
        self.img_dir = os.path.join(root_dir, 'filtered_smile_img')
        flip_img_dir = os.path.join(root_dir, 'flip_filtered_smile_img')
        name_list = []
        for img_name in os.listdir(self.img_dir):
            name_list.append(os.path.join(self.img_dir, img_name))
            name_list.append(os.path.join(flip_img_dir, 'flip_'+img_name))
        self.list_ids = name_list

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        
        img_name = self.list_ids[index]
        mask_name = img_name
        img = Image.open(img_name)
        img = np.array(img).astype(np.float32) / 255 * 2 - 1

        kernel_size = np.random.randint(5, 15)
        # print(self.src_mask_dir, mask_name)
        fg = np.ones((256, 256))
        # fg = (fg  > 0.4).astype(np.float32)

        mask = np.ones((256, 256))

        transition = mask - fg

        mask = mask.astype(np.float32)
        input_semantic = np.zeros((256, 256, 2), dtype=np.float32)
        input_semantic[..., 0] = fg
        input_semantic[..., 1] = transition

        input_semantic = input_semantic.transpose(2, 0, 1)
        bg = (1 - mask)[..., None] * img

        return {
            'img': img.transpose(2, 0, 1),
            'bg': bg.transpose(2, 0, 1),
            'mask': mask.astype(np.float32)[None],
            'input_semantic': input_semantic,
        }

class less_conditional(Dataset):
    def __init__(self):
        list_ids = []
        mask_ids = []
        limg = os.listdir('/home/vsfh/dataset/smile')
        for imgn in limg:
            list_ids.append(os.path.join('/home/vsfh/dataset/smile', imgn))
            mask_ids.append(os.path.join('/home/vsfh/dataset/filtered_smile_mask', imgn))

            # list_ids.append(os.path.join('/home/vsfh/dataset/flip/down', imgn))
            # mask_ids.append(os.path.join('/home/vsfh/dataset/flip/down_mask', imgn))

        self.list_ids = list_ids
        self.mask_ids = mask_ids

    def __len__(self):
        return len(self.list_ids)-1

    def __getitem__(self, index):
        img_name = self.list_ids[index]
        mask_name = self.mask_ids[index]
        img = Image.open(img_name)
        img = np.array(img).astype(np.float32) / 255 * 2 - 1

        kernel_size = 4
        fg = Image.open(mask_name)
        # print(self.src_mask_dir, mask_name)
        fg = np.array(fg)[:, :, 0]/255
        # fg[:40, :] = 1
        # fg[216:, :] = 1
        # fg = (fg  > 0.4).astype(np.float32)

        mask = cv2.dilate(fg, kernel=np.ones((kernel_size, kernel_size)))

        transition = mask - fg

        mask = mask.astype(np.float32)
        input_semantic = np.zeros((256, 256, 2), dtype=np.float32)
        input_semantic[..., 0] = fg
        input_semantic[..., 1] = transition

        input_semantic = input_semantic.transpose(2, 0, 1)
        bg = (1 - mask)[..., None] * img

        return {
            'img': img.transpose(2, 0, 1),
            'bg': bg.transpose(2, 0, 1),
            'mask': mask.astype(np.float32)[None],
            'input_semantic': input_semantic,
        }

class encoder_less_conditional(Dataset):
    def __init__(self):
        root_dir = '/home/vsfh/dataset/teeth_align'
        self.img_dir = os.path.join(root_dir, 'up_unalign_teeth')
        self.mask_dir = os.path.join(root_dir, 'up_unalign_teeth_mask')

        self.list_ids = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        img_name = self.list_ids[index]
        mask_name = img_name
        img = Image.open(os.path.join(self.img_dir, img_name))
        img = np.array(img).astype(np.float32) / 255 * 2 - 1

        kernel_size = np.random.randint(5, 15)
        fg = Image.open(os.path.join(self.mask_dir, mask_name))
        fg = np.array(fg)[:, :, 0]/255
        # fg[:40, :] = 1
        # fg[216:, :] = 1
        mask = cv2.dilate(fg, kernel=np.ones((kernel_size, kernel_size)))
        transition = mask - fg
        mask = mask.astype(np.float32)
        input_semantic = np.zeros((256, 256, 2), dtype=np.float32)
        input_semantic[..., 0] = fg
        input_semantic[..., 1] = transition
        input_semantic = input_semantic.transpose(2, 0, 1)
        bg = (1 - mask)[..., None] * img
        
        fg2 = np.ones((256, 256))
        mask2 = np.ones((256, 256))
        transition2 = mask2 - fg2
        mask2 = mask2.astype(np.float32)
        input_semantic2 = np.zeros((256, 256, 2), dtype=np.float32)
        input_semantic2[..., 0] = fg2
        input_semantic2[..., 1] = transition2
        input_semantic2 = input_semantic2.transpose(2, 0, 1)
        bg2 = (1 - mask2)[..., None] * img

        return {
            'img': img.transpose(2, 0, 1),
            'bg': bg.transpose(2, 0, 1),
            'mask': mask.astype(np.float32)[None],
            'bg2': bg2.transpose(2, 0, 1),
            'mask2': mask2.astype(np.float32)[None],
            'input_semantic2': input_semantic2,
            
        }
class AlignedDataset(Dataset):
    def __init__(self):
        root_dir = r'D:\sfh\dataset\teeth_img'
        self.img_dir = os.path.join(root_dir, 'up_unaligned_teeth')
        self.mask_dir = os.path.join(root_dir, 'up_unaligned_teeth_mask')

        self.list_ids = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        img_name = self.list_ids[index]
        mask_name = img_name
        img = Image.open(os.path.join(self.img_dir, img_name))
        img = np.array(img).astype(np.float32) / 255 * 2 - 1

        kernel_size = np.random.randint(5, 15)
        fg = Image.open(os.path.join(self.mask_dir, mask_name))
        # print(self.src_mask_dir, mask_name)
        fg = np.array(fg)[:,:,0]/255
        # fg = (fg  > 0.4).astype(np.float32)

        mask = cv2.dilate(fg, kernel=np.ones((kernel_size, kernel_size)))

        transition = mask - fg

        mask = mask.astype(np.float32)
        input_semantic = np.zeros((256, 256, 2), dtype=np.float32)
        input_semantic[..., 0] = fg
        input_semantic[..., 1] = transition

        input_semantic = input_semantic.transpose(2, 0, 1)
        bg = (1 - mask)[..., None] * img

        return {
            'img': img.transpose(2, 0, 1),
            'bg': bg.transpose(2, 0, 1),
            'mask': mask.astype(np.float32)[None],
            'input_semantic': input_semantic,
        }

class SmileDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )


        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'image_{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            buffer = BytesIO(img_bytes)
            img = Image.open(buffer)

            key = f'segmap1_{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            buffer = BytesIO(img_bytes)
            segmap1 = np.array(Image.open(buffer))

            key = f'segmap2_{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            buffer = BytesIO(img_bytes)
            segmap2 = np.array(Image.open(buffer))

        img = np.array(img).astype(np.float32) / 255 * 2 - 1
        segmap = np.concatenate([segmap1[..., ::-1], segmap2[..., 1:]], axis=-1)
        segmap = (segmap > 255 * 0.3).astype(np.float32)

        segmap[..., 1] = skeletonize(segmap[..., 1])
        segmap[..., 1] = cv2.dilate(segmap[..., 1], np.ones((3,3)))


        segmap[..., 3] = skeletonize(segmap[..., 3])
        segmap[..., 3] = cv2.dilate(segmap[..., 3], np.ones((3,3)))

        segmap[..., 4] = skeletonize(segmap[..., 4])
        segmap[..., 4] = cv2.dilate(segmap[..., 4], np.ones((3,3)))

        # segmap = np.array(segmap).astype(np.float32) / 255

        kernel_size = np.random.randint(5, 15)

        fg = (segmap[..., 0] > 0.4).astype(np.float32)
        mask = cv2.dilate(fg, kernel=np.ones((kernel_size, kernel_size)))

        # cv2.imshow()
        transition = mask - fg

        # for i in range(segmap.shape[-1]):
        #     cv2.imshow(str(i), segmap[..., i])
        # cv2.imshow('o', segmap[..., 0])
        # cv2.imshow('fg', fg)
        # cv2.imshow('tran', transition)
        # cv2.waitKey()

        # input_semantic = np.concatenate([segmap, transition[..., None]], axis=-1)
        input_semantic = segmap

        teeth_mask = input_semantic[..., 2]
        ksize = kernel_size + 3
        teeth_mask = cv2.dilate(teeth_mask, np.ones((ksize, ksize)))

        mask = np.logical_or(teeth_mask, mask)
        mask = mask.astype(np.float32)
        # cv2.imshow('mask', teeth_mask)

        # cv2.imshow('tmp1', input_semantic[..., :3])
        input_semantic[..., 0] = mask
        # cv2.imshow('tmp2', input_semantic[..., :3])


        # input_semantic[..., 0] = (input_semantic[..., 0] - input_semantic[..., 2]) * input_semantic[..., 0]


        input_semantic = input_semantic.transpose(2, 0, 1)
        bg = (1 - mask)[..., None] * img

        # for i, o in enumerate(input_semantic):
        #     cv2.imshow(str(i), o)
        # cv2.waitKey()

        return {
            'img': img.transpose(2, 0, 1),
            'bg': bg.transpose(2, 0, 1),
            'mask': mask.astype(np.float32)[None],
            # 'teeth'
            'input_semantic': input_semantic,
            # 'bg': bg,
        }

class SimulationDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )


        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'image_{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            buffer = BytesIO(img_bytes)
            img = Image.open(buffer)

            key = f'segmap1_{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            buffer = BytesIO(img_bytes)
            segmap1 = np.array(Image.open(buffer))

        img = np.array(img).astype(np.float32) / 255 * 2 - 1
        segmap = segmap1[..., ::-1]
        segmap = (segmap > 255 * 0.3).astype(np.float32)

        # segmap = np.array(segmap).astype(np.float32) / 255

        kernel_size = np.random.randint(5, 15)

        # cv2.imshow('seg', segmap)
        # cv2.waitKey()
        fg = (segmap[..., 0] > 0.4).astype(np.float32)
        mask = cv2.dilate(fg, kernel=np.ones((kernel_size, kernel_size)))

        transition = mask - fg


        mask = mask.astype(np.float32)
        input_semantic = np.zeros((256,256, 2), dtype=np.float32)
        input_semantic[..., 0] = mask
        input_semantic[..., 1] = transition

        input_semantic = input_semantic.transpose(2, 0, 1)
        bg = (1 - mask)[..., None] * img


        return {
            'img': img.transpose(2, 0, 1),
            'bg': bg.transpose(2, 0, 1),
            'mask': mask.astype(np.float32)[None],
            # 'teeth'
            'input_semantic': input_semantic,
            # 'bg': bg,
        }

class fuck(Dataset):
    def __init__(self, mode='decoder'):
        if mode == 'encoder':
            self.all_files = glob.glob('/mnt/share/shenfeihong/data/smile_/seg_6000/*/mouth.jpg', recursive=False)[10:]
            self.transform = transforms.Compose(
                                [
                                    transforms.Resize([256,256]),
                                    transforms.ToTensor(),
                                    # transforms.RandomHorizontalFlip(p=0.5)
                                    
                                ]
                        )       
        else:
            self.path = '/mnt/share/shenfeihong/data/smile_/mouth_seg/'
            self.all_files = glob.glob('/mnt/share/shenfeihong/data/smile_/2022-11-30-tianshi/seg/*/mouth.png')
                                # glob.glob('/mnt/share/shenfeihong/data/smile_/seg_6000/*/mouth.jpg')

            self.transform = transforms.Compose(
                                [
                                    transforms.Resize([256,256]),
                                    transforms.ToTensor()
                                    
                                ]
                        )
        print('total image:', len(self.all_files))
        self.mode = mode
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        frame_file = self.all_files[index]
        f_name = frame_file.split('/')[-1]

        ske_file = frame_file.replace(f_name,'mask.png')
        inner = np.random.randint(5)
        
        frame = Image.open(frame_file)
        mask = Image.open(ske_file)
        
        mask = np.array(mask)/255
        if len(mask.shape) == 3:
            mask = mask[...,0]

        # mask = cv2.dilate(mask, kernel=np.ones((30+inner, 30+inner)))-cv2.dilate(mask, kernel=np.ones((inner, inner)))
        # big_mask = cv2.dilate(mask, kernel=np.ones((inner, inner)))
        # mask = mask.astype(np.float32)[None]
        # big_mask = big_mask.astype(np.float32)[None]
        cir_mask = cv2.dilate(mask, kernel=np.ones((20+inner, 20+inner)))-cv2.dilate(mask, kernel=np.ones((inner, inner)))
        cir_mask = cir_mask.astype(np.float32)[None]
        
        big_mask = cv2.dilate(mask, kernel=np.ones((inner, inner)))
        big_mask = mask.astype(np.float32)[None]
        
        ##edge
        fdir = frame_file.split('/')[-3]

        edge = Image.open(frame_file.replace(f_name,'edge.png'))

        edge = np.array(edge)/255
        if len(edge.shape) == 3:
            edge = edge[...,0]

        edge = edge.astype(np.float32)[None]
                
        img = self.transform(frame)*2-1
        if np.random.randint(2)>0:
            img = flip(img,2)
            cir_mask = cir_mask[:,:,::-1].copy()
            big_mask = big_mask[:,:,::-1].copy()
            edge = edge[:,:,::-1].copy()
        return {'images': img, 'mask':cir_mask, 'big_mask':big_mask, 'edge':edge}

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

from natsort import natsorted
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

        return {'mask': mask, 'mouth':img, 'label':label, 'up_edge':up_edge}
       
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
    ds = fuck()

    for batch in ds:
        images = batch['images']
        mask = batch['mask']
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
        cv2.imshow('img', ((images+1)/2).numpy().transpose(1,2,0))
        cv2.waitKey(0)


