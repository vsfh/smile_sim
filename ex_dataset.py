from torch.utils.data import Dataset
# from PIL import Image
# from utils import data_utils
import os
import cv2
import numpy as np

class RandomPerspective:

    def __init__(self,
                 degrees=0.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.0,
                 perspective=0.0,
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

    def affine_transform(self, img, border):
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
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s

    def __call__(self, labels):
        """
        Affine images and targets.

        Args:
            labels (dict): a dict of `bboxes`, `segments`, `keypoints`.
        """
        if self.pre_transform and 'mosaic_border' not in labels:
            labels = self.pre_transform(labels)
        labels.pop('ratio_pad', None)  # do not need ratio pad

        img = labels['img']
        cls = labels['cls']
        instances = labels.pop('instances')
        # Make sure the coord formats are right
        instances.convert_bbox(format='xyxy')
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop('mosaic_border', self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format='xyxy', normalized=False)
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(box1=instances.bboxes.T,
                                box2=new_instances.bboxes.T,
                                area_thr=0.01 if len(segments) else 0.10)
        labels['instances'] = new_instances[i]
        labels['cls'] = cls[i]
        labels['img'] = img
        labels['resized_shape'] = img.shape[:2]
        return labels



class ImagesDataset(Dataset):
    def __init__(self, mode='train'):
        self.path = '/ssd/gregory/smile/out/'
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
        edge = cv2.imread(os.path.join(img_folder, 'edge.png'))
        
        
        im = self.preprocess(img)
        mk = self.preprocess(mask)
        teeth_3d = self.preprocess(teeth_3d)
        ed = self.preprocess(edge)
        
        cond = ed*mk+im*(1-mk)
        
        return {'images': im, 'cond':cond}
        
    def preprocess(self, img):
        img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        if len(img_resize.shape)==2:
            img_resize = img_resize[:,:,None].repeat(1,1,3)
        im = np.ascontiguousarray(img_resize.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = im / 255.0  # 0-255 to 0.0-1.0
        return im