import torch
import torch.nn as nn
import numpy as np
import cv2
import pyutils
from pytorch3d.transforms.rotation_conversions import *

import time
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
import os
import glob
from natsort import natsorted
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import imageio
# rendering components
from pytorch3d.renderer import (
	RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
	PerspectiveCameras, SoftPhongShader,HardPhongShader, TexturesVertex, PointLights,SoftSilhouetteShader
)
import onnxruntime
from tid_models import get_tid, get_yolo_tid

onnx_opt_dict = {
    'smile_sim_lip_preserve-yolov5':{'input':['images'],'output':['output'],'path':'/mnt/d/triton/backup_model/smile_sim_lip_preserve-yolov5/1'},
    'smile_sim_lip_preserve-edge_net':{'input':['data'],'output':['output'],'path':'/mnt/d/triton/backup_model/smile_sim_lip_preserve-edge_net/1'},
    'cls-ensemble':{'input':['images'],'output':['output'],'path':'/mnt/d/triton/backup_model/cls-yolov5s/1'},
    'new_smile_wo_edge_gan':{'input':['input_image','mask'],'output':['align_img'],'path':'/mnt/d/triton/backup_model/new_smile_wo_edge_gan/1'},
}
onnx_sess_dict = {}
for k,v in onnx_opt_dict.items():
    onnx_sess_dict[k] = onnxruntime.InferenceSession(os.path.join(v['path'],'model.onnx'),providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    
def onnx_infer(data, sess_name):
    input = {name: data[i] for i,name in enumerate(onnx_opt_dict[sess_name]['input'])}
    output_names = onnx_opt_dict[sess_name]['output']
    if len(output_names)==1:
        output = onnx_sess_dict[sess_name].run([], input)[0]
        return output
    else:
        print('error', sess_name)

def _seg_tid(image):
    teeth_model = get_yolo_tid('/mnt/d/triton/backup_model', backend='native')
    tid = get_tid(teeth_model=teeth_model, img=image)

    return tid

def _find_objs(image):
    yolo_input_shape = (640, 640)
    original_h, original_w = image.shape[:2]
    resized_image, meta = pyutils.resize_and_pad(image, yolo_input_shape)
    offsets = meta['offsets']
    scale = meta['scale']
    input_imgs = pyutils.normalize_img(resized_image)
    output = onnx_infer([input_imgs],'smile_sim_lip_preserve-yolov5', )
    output = output[0]
    xywh = output[:, :4]
    probs = output[:, 4:5] * output[:, 5:]

    objs = []
    num_class = probs.shape[-1]

    for i in range(num_class):
        p = probs[:, i]
        if p.max() < 0.04:
            a = np.array([0,0,0,0])
            objs.append(a)
            continue
        idx = p.argmax()

        x, y, w, h = xywh[idx]
        coords = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

        coords[[0, 2]] -= offsets[0]
        coords[[1, 3]] -= offsets[1]
        coords /= scale

        coords = pyutils.loose_bbox(coords, (original_w, original_h))
        objs.append(coords)
    objs = np.array(objs, dtype=int)
    return objs

def _seg_mouth(image):
    seg_input_shape = (256, 256)
    resized_image, _ = pyutils.resize_and_pad(image, seg_input_shape)
    input_imgs = pyutils.normalize_img(resized_image)
    output = onnx_infer([input_imgs], 'smile_sim_lip_preserve-edge_net')
    output = np.transpose(output[0], (1, 2, 0))
    output = pyutils.sigmoid(output)
    return output

def apply_style(img, mean, std, mask=None, depth=None):
    if len(mask.shape) == 2:
        mask = mask[..., None]
    if mask.shape[2] == 3:
        mask = mask[..., :1]
    mask = mask.astype(np.uint8)

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype("float32")

    src_mean, src_std = cv2.meanStdDev(img_lab, mask=mask)
    img_lab = (img_lab - src_mean.squeeze()) / src_std.squeeze() * std.squeeze() + mean.squeeze()
    img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    img_rgb = img_rgb.astype(np.float32) * (depth[...,None] * 0.3 + 0.7)* (depth[...,None] * 0.3 + 0.7)

    mask = mask.astype(np.float32)

    smooth_mask = cv2.blur(mask, (5, 5))
    smooth_mask = smooth_mask[..., None]

    result = img_rgb * smooth_mask + (1 - smooth_mask) * img
    result = result.astype(np.uint8)

    return result

def detection_and_segmentation(face_img):
    height, width = face_img.shape[:2]

    # step 1. find mouth obj
    objs = _find_objs(face_img)

    mouth_objs = objs[2]
    x1, y1, x2, y2 = mouth_objs
    if x1==x2 and y1==y2:
        raise Exception('not smile image')

    w, h = (x2 - x1), (y2 - y1)
    
    half = max(w, h) * 1.1 / 2
    # half = max(w, h) / 2
    

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    x1, y1, x2, y2 = pyutils.loose_bbox([x1, y1, x2, y2], (width, height))
    x, y = int(x1 * 128 / half), int(y1 * 128 / half) + 2

    template = cv2.resize(face_img, (int(width * 128 / half), int(height * 128 / half)), cv2.INTER_AREA)
    mouth = template[y: y + 256, x: x + 256]

    # tid, upper, lower = _seg_tid(mouth)
       
    # step 2. edgenet seg mouth
    seg_result = _seg_mouth(mouth)

    mouth_mask = (seg_result[..., 0] > 0.6).astype(np.float32)
    teeth_mask = (seg_result[..., 4] > 0.6).astype(np.uint8)
    edge = (seg_result[..., 1] > 0.6).astype(np.float32)
    up_edge = (seg_result[..., 3] > 0.6).astype(np.float32)
    # for i in range(seg_result.shape[-1]):
    #     a = (seg_result[..., i] > 0.6).astype(np.uint8)
    #     cv2.imshow('img', a*255)
    #     cv2.waitKey(0)
    
    # heatmap = tid
    
    contours, _ = cv2.findContours(mouth_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours)>1:
        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        idx = np.argmax(np.array(area))
        # print(idx)
        for k in range(len(contours)):
            if k!= idx:
                mouth_mask = cv2.drawContours(mouth_mask, contours, k, 0, cv2.FILLED)
                
    return {'mouth_mask': mouth_mask,
            'teeth_mask': teeth_mask,
            'edge': edge,
            # 'heatmap': heatmap,
            'up_mask': up_edge,
            'mouth_img': mouth,
            'face_img': template,
            'position': (y,x),
            'width_height': (width, height),
            # 'upper_lower': (upper, lower)
            }


def fitting(seg_res, tooth_dict, step_list, device='cpu'):
    class SoftEdgeShader(nn.Module):
        def __init__(self, device="cpu", blend_params=None, zfar=200, znear=1):
            super().__init__()
            self.blend_params = blend_params if blend_params is not None else BlendParams()
            self.zfar = zfar
            self.znear = znear

        def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
            blend_params = kwargs.get("blend_params", self.blend_params)
            extra_mask = kwargs.get("extra_mask", None)

            mask = fragments.pix_to_face >= 0
            if extra_mask is not None:
                mask = mask * extra_mask

            prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
            prob_map = torch.sum(prob_map, -1)
            alpha = torch.prod((1.0 - prob_map), dim=-1, keepdim=True)
            alpha = 1 - alpha    
                    
            zbuf = fragments.zbuf[...,-1]
            zbuf[zbuf==-1] = 1e10
            zbuf = torch.cat((torch.ones_like(zbuf[0][None])*1e10,zbuf),0)
            zbuf_mask = torch.argmin(zbuf, 0, keepdim=True)
            
            for i in range(len(prob_map)):
                prob_map[i] = prob_map[i]*(zbuf_mask[0]==i+1)
            
            
            return prob_map, 0

    class EdgeShader(nn.Module):
        def forward(self, fragments, meshes, **kwargs):
            bg_value = 1e6
            zbuf = fragments.zbuf
            N, H, W, K = zbuf.shape

            zbuf[zbuf == -1] = bg_value

            depth, _ = torch.min(zbuf[:-1], dim=0)
            max_depth, min_depth = depth[depth != bg_value].max(), depth[depth != bg_value].min()
            new_depth = 1 - (depth -min_depth) / (max_depth - min_depth)
            new_depth[depth == bg_value] = 0


            bg = torch.ones((1, H, W), device=zbuf.device) * (bg_value - 1)
            zbuf_with_bg = torch.cat([bg, zbuf[..., 0]], dim=0)
            teeth = torch.argmin(zbuf_with_bg, dim=0)

            return teeth, new_depth.cpu().numpy()

    class Model(nn.Module):
        def __init__(self, meshes, image_ref, teeth_region, focal_length, axis_angle, translation,
                    raster_settings):
            super().__init__()
            self.up_mesh, self.down_mesh = meshes
            self.device = self.up_mesh.device

            image_ref = torch.from_numpy(image_ref[np.newaxis, ...].astype(np.float32))
            self.register_buffer('image_ref', image_ref)

            mouth_mask, up_mask = teeth_region
            mouth_mask = torch.from_numpy(mouth_mask[np.newaxis, ...].astype(np.float32))
            up_mask = torch.from_numpy(up_mask[np.newaxis, ...].astype(np.float32))
            
            self.register_buffer('mouth_mask', mouth_mask)
            self.register_buffer('up_mask', up_mask)
            
            self.focal_length = nn.Parameter(torch.tensor(focal_length, dtype=torch.float32, device=self.device))
            
            self.angle_x = nn.Parameter(torch.tensor(axis_angle[:1], dtype=torch.float32, device=self.device))
            self.angle_y = nn.Parameter(torch.tensor(axis_angle[1:2], dtype=torch.float32, device=self.device))
            self.angle_z = nn.Parameter(torch.tensor(axis_angle[2:3], dtype=torch.float32, device=self.device))

            self.x = nn.Parameter(torch.tensor((translation[0],), dtype=torch.float32, device=self.device))
            self.y = nn.Parameter(torch.tensor((translation[1],), dtype=torch.float32, device=self.device))
            self.z = nn.Parameter(torch.tensor((translation[2],), dtype=torch.float32, device=self.device))

            self.dist_x = nn.Parameter(torch.tensor((-0,), dtype=torch.float32, device=self.device))
            self.dist_y = nn.Parameter(torch.tensor((-0,), dtype=torch.float32, device=self.device))
            self.dist_z = nn.Parameter(torch.tensor((2,), dtype=torch.float32, device=self.device))

            self.cameras = PerspectiveCameras(device=self.device, focal_length=self.focal_length)

            self.renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftEdgeShader(
                    blend_params=BlendParams(sigma=1e-6, gamma=1e-2, background_color=(0., 0., 0.)))
            )
            # self.seg_model = torch.jit.load('traced_bert.pt')
            # self.seg_model.to(self.device)
            # for param in self.seg_model.parameters():
            #     param.requires_grad = False

        def render(self):
            with torch.no_grad():
                axis_angles = torch.cat([self.angle_x, self.angle_y, self.angle_z])
                R = axis_angle_to_matrix(axis_angles[None, :])

                translation = torch.cat([self.x, self.y, self.z])
                T = translation[None, :]
                image = self.renderer(meshes_world=self.meshes, R=R, T=T, extra_mask=None)
            return image

        def forward(self):
            axis_angles = torch.cat([self.angle_x, self.angle_y, self.angle_z])
            R = axis_angle_to_matrix(axis_angles[None, :])

            translation = torch.cat([self.x, self.y, self.z])
            T = translation[None, :]

            down_offset = torch.cat([self.dist_x, self.dist_y, self.dist_z])
            meshes = join_meshes_as_batch([self.up_mesh, self.down_mesh.offset_verts(down_offset)], include_textures=False)
            
            image, zbuf_mask = self.renderer(meshes_world=meshes.clone(), R=R, T=T, extra_mask=None)  
            # image = torch.where(image != 0, torch.ones_like(image), image)
            image = image * self.mouth_mask
                      
            im = image.view(image.shape[0],1,*image.shape[1:3])
            
            im = pyutils.erode(pyutils.dilate(im))
            im = pyutils.dilate(im-pyutils.erode(im))
            im = torch.sum(im, 0).clip(0,1)
            
            label = pyutils.dilate(self.image_ref[None])
            pred = pyutils.dilate(im)

            loss = torch.sum(((pred - label)) ** 2)
            return loss, im, R, T, down_offset
    
    step1 = step_list['step_0']
    teeth = pyutils.load_teeth(tooth_dict, type='tooth', half=True, sample=True, voxel_size=0.8)
    
    mouth_mask = seg_res['mouth_mask']
    teeth_mask = seg_res['teeth_mask']
    # upper_lower = seg_res['upper_lower']
    up_mask = seg_res['up_mask']

    closed_teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE,
                                         kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)))
    closed_up_teeth_mask = cv2.morphologyEx(up_mask, cv2.MORPH_CLOSE,
                                         kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))



    visible_area = mouth_mask

    image_ref = seg_res['edge']
    up_mesh = pyutils.apply_step(teeth, step1, mode='up', add=False, num_teeth=3)
    up_tensor = pyutils.meshes_to_tensor(up_mesh, device=device)
    down_mesh = pyutils.apply_step(teeth, step1, mode='down', add=False, num_teeth=3)
    down_tensor = pyutils.meshes_to_tensor(down_mesh, device=device)
    up_bbox = up_mesh[0].get_axis_aligned_bounding_box()
    ref_y, ref_z = up_bbox.min_bound[1], up_bbox.max_bound[2]
    ref = -ref_y * np.sin(np.deg2rad(10)) + ref_z * np.cos(np.deg2rad(10))

    index = np.argwhere(np.max(up_mask[:, 100:150], axis=1) > 0)
    if len(index)==0:
        mouth_mid = 128
    else:
        mouth_mid = np.max(index)

    focal_length = 13
    init_z = 475
    scale = focal_length * 128 / init_z
    init_y = (128 - mouth_mid) / scale + ref

    translation = (0, init_y, init_z)
    axis_angle = [np.deg2rad(-80), np.deg2rad(0), 0]

    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=2e-3,
        faces_per_pixel=25,
        perspective_correct=False,
        cull_backfaces=True
    )

    model = Model(meshes=[up_tensor, down_tensor], image_ref=image_ref,
                  teeth_region=(closed_teeth_mask, closed_up_teeth_mask),
                  focal_length=focal_length,
                  translation=translation,
                  axis_angle=axis_angle,
                  raster_settings=raster_settings).to(device)

    params = [{'params': model.angle_x, 'lr': 1e-7, 'name': 'angle_x'},
              {'params': model.angle_y, 'lr': 1e-7, 'name': 'angle_y'},
              {'params': model.angle_z, 'lr': 1e-7, 'name': 'angle_z'},

              {'params': model.x, 'lr': 3e-4, 'name': 'x'},
              {'params': model.y, 'lr': 3e-4, 'name': 'y'},
              {'params': model.z, 'lr': 3e-4, 'name': 'z'},

              {'params': model.dist_x, 'lr': 3e-7, 'name': 'dist_y'},
              {'params': model.dist_y, 'lr': 3e-7, 'name': 'dist_y'},
              {'params': model.dist_z, 'lr': 3e-7, 'name': 'dist_y'},

              {'params': model.focal_length, 'lr': 1e-5, 'name': 'focal_length'}]

    optimizer = torch.optim.SGD(
        params,
        lr=1e-3, momentum=0.1)

    min_loss = np.Inf
    gif_images = []
    start = time.time()
    for i in range(1000):
        optimizer.zero_grad()
        loss, teeth_mask, R, T, dist = model()
        # print(loss, R, T, dist)
        # if i % 10 == 0:
        #     teeth_mask = teeth_mask.detach().cpu().numpy()
        #     diff = np.zeros((256, 256, 3), dtype=np.float32)
        #     diff[..., 0] = teeth_mask
        #     diff[..., 1] = image_ref
        #     gif_images.append((diff * 255).astype(np.uint8))
        
        if loss.item() < min_loss:
            best_opts = {
                'focal_length': model.focal_length.detach().clone(),
                'R': R.detach().clone(),
                'T': T.detach().clone(),
                'dist': dist.detach().clone(),
            }
            min_loss = loss.item()
            min_step = i
            
        if (i - min_step) > 100:
            break
        loss.backward()
        optimizer.step()
    # imageio.mimsave('optimization.gif', gif_images, duration=100)
    # print(loss.item())
    if loss.item()>1200 or up_mask.sum()<1200:
        return 0,0,0,False
    opt_cameras = PerspectiveCameras(device=device, focal_length=best_opts['focal_length'])
    lights = PointLights(device=device, ambient_color=((0.9, 0.9, 0.9),), location=[[2.0, -60.0, -12.0]])
    edge_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=opt_cameras,
            raster_settings=RasterizationSettings(
                image_size=256,
                faces_per_pixel=1,
                perspective_correct=True,
                cull_backfaces=True,
            )
        ),
        # shader=EdgeShader()
        shader=HardPhongShader(device=device, cameras=opt_cameras, lights=lights,blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0., 0., 0.)))
    )

    # teeth_root = utils.load_teeth(tooth_root_dict, type='root', half=False, sample=False)
    edge_dict = {}
    depth_dict = {}
    id_dict = {}
    start = time.time()
    for step_idx in step_list.keys():
        step_array = step_list[step_idx]
        # step = utils.load_step_file(step_array)
        step = step_array
        up_mesh = pyutils.apply_step(teeth, step, mode='up', add=False, num_teeth=6)
        up_tensor = pyutils.meshes_to_tensor(up_mesh,'scene', device=device)
        down_mesh = pyutils.apply_step(teeth, step, mode='down', add=False, num_teeth=6)
        down_tensor = pyutils.meshes_to_tensor(down_mesh,'scene', device=device)

        mid = len(up_mesh)

        with torch.no_grad():
            R = best_opts['R']
            T = best_opts['T']
            dist = best_opts['dist']
            dist[-1] = dist[-1]
            
            teeth_mesh = join_meshes_as_scene([up_tensor, down_tensor.offset_verts(dist)],
                                              include_textures=True)

            out_im = edge_renderer(meshes_world=teeth_mesh, R=R, T=T)
            out_im = (255*out_im[0,:,:,:3]/out_im[0,:,:,:3].max()).detach().cpu().numpy().astype(np.uint8)

        depth_dict[step_idx] = out_im
        
        # id_dict[step_idx] = down_edge
    return edge_dict, depth_dict, id_dict, True

def _gan(inputs, network_name, tinf=None):
    output = onnx_infer(inputs, network_name, )
    output = output[0].transpose(1, 2, 0)
    output = (output + 1.0) / 2.0
    return output

def smile_generation_based_on_edge(seg_res, edge_list, depth_dict, id_dict):
    ori_mask = seg_res['mouth_mask']
    # ori_mask = cv2.dilate(ori_mask, np.ones((3,3)))
    mouth_img = seg_res['mouth_img']
    face_img = seg_res['face_img']
    wh = seg_res['width_height']
    xy = seg_res['position']
    kernel = np.ones((4,4),np.uint8)

    mouth_lab = cv2.cvtColor(mouth_img, cv2.COLOR_RGB2LAB)
    ori_teeth_mask = cv2.erode(seg_res['teeth_mask'], kernel)
    target_mean, target_std = cv2.meanStdDev(mouth_lab, mask=ori_teeth_mask)
    smile_img_list = {}
    mouth_img = mouth_img/255
    
    for edge_idx in edge_list.keys():

        up_edge = edge_list[edge_idx]
        down_edge = id_dict[edge_idx]
        
        edge = np.logical_or(up_edge, down_edge).astype(np.float32)
        
        depth_ori = depth_dict[edge_idx]
        depth = depth_ori.copy()
        depth[depth_ori>0] = 1
        depth[ori_mask==0] = 0

        edge[edge>0] = 1
        up_edge[up_edge>0] = 1
        
        contour1, _ = cv2.findContours(up_edge.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour2, _ = cv2.findContours(down_edge.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tmask = np.zeros_like(edge)
        cv2.drawContours(tmask, contour1, -1, (1), thickness=cv2.FILLED)
        cv2.drawContours(tmask, contour2, -1, (1), thickness=cv2.FILLED)
        
        edge = edge[...,None].repeat(3,2)
        up_edge = up_edge[...,None].repeat(3,2)
        
        mask = ori_mask[...,None].repeat(3,2)
        tmask = tmask[...,None].repeat(3,2)

        cond = edge*mask*0.1 + up_edge*mask*0.5 + tmask*mask*(1-edge) + mouth_img*(1-mask)
        network_name = 'smile_sim_lip_preserve-up_net'
        aligned_mouth = _gan([pyutils.normalize_img(cond)], network_name, )
        aligned_mouth = aligned_mouth.clip(0,1)*255
        aligned_mouth = aligned_mouth.astype(np.uint8)
        
        depth_ori = cv2.dilate(depth_ori.repeat(3,2), kernel)
        
        aligned_mouth = apply_style(aligned_mouth.copy(), target_mean, target_std, mask=depth[...,0], depth=depth_ori[...,0])
        
        aligned_mouth = mouth_img*(1-mask)*255 + aligned_mouth*mask
        local_face = face_img.copy()
        local_face[xy[0]: xy[0] + 256, xy[1]: xy[1] + 256] = aligned_mouth
        local_face = cv2.resize(local_face, wh)
        smile_img_list[edge_idx] = local_face
    return smile_img_list

def blend_images(image_a, image_b, opacity):
    blended_image = cv2.addWeighted(image_a, 1 - opacity, image_b, opacity, 0)
    return blended_image

class predictor(object):
    def __init__(self) -> None:
        pass
    def predict(self, face_img: np.array, tooth_dict, step_list):
        seg_res = detection_and_segmentation(face_img)
        edge_dict, depth_dict, id_dict, yep = fitting(seg_res, tooth_dict, step_list, device='cuda')
        if not yep:
            return None
        a = depth_dict['step_0']
        b = seg_res['mouth_img'][...,::-1]
        c = seg_res['mouth_mask']
        d = blend_images(a,b,0.7)
        e = seg_res['edge']
        f = seg_res['up_mask']

        return a,b,c,d,e,f

        smile_img_list = smile_generation_based_on_edge(seg_res, edge_dict, depth_dict, id_dict)
        torch.cuda.empty_cache()
        return smile_img_list, edge_dict
  
  
def visualize_teeth(face_img, tooth_dict, step_list):
    import trimesh
    seg_res = detection_and_segmentation(face_img)
    step1 = pyutils.load_step_file(step_list['step_0'])
    upper_lower = seg_res['upper_lower']
    up_mesh = pyutils.trimesh_load_apply(tooth_dict, step1)
    up_mesh[11].show()
import natsort
def test():
    smile = predictor()
    path = '/mnt/d/data/smile/Teeth_simulation_10K/'
    with open('/mnt/d/data/smile/error.txt', 'r') as f:
        error_list = [line.strip() for line in f.readlines()]
    for case in tqdm(natsort.natsorted(os.listdir(path)[830:])):
        # case = 'C01002721541'
        # case = '0b01e19ba8f3daa5aed0653dd253a78e'
        img_folder = os.path.join(path,case)
        if not os.path.exists(os.path.join(img_folder, '微笑像.jpg')) or case in error_list:
            continue
        try:
            face_img = np.array(Image.open(os.path.join(img_folder, '微笑像.jpg')))
        except:
            print('img', case)
        tooth_dict = {int(os.path.basename(p).split('/')[-1][:2]): trimesh.load(p) for p in glob.glob(os.path.join(img_folder, 'models', '*._Root.stl'))}
        steps_dict = {}
        step_one_dict = {}
        
        step_list = natsorted(glob.glob(img_folder+'/models/*.txt'))
        for arr in np.loadtxt(step_list[0]):
            trans = np.eye(4,4)
            trans[:3,3] = arr[1:4]
            trans[:3,:3] = R.from_quat(arr[-4:]).as_matrix()
            step_one_dict[str(int(arr[0]))] = trans
        steps_dict['step_0'] = step_one_dict
        try:
            res = smile.predict(face_img, tooth_dict, steps_dict)
            if res is None:
                continue
            a,b,c,d,e,f = res
            save_path = os.path.join('/mnt/d/data/smile/out',case)
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, 'teeth_3d.png'), a)
            cv2.imwrite(os.path.join(save_path, 'mouth.png'), b)
            cv2.imwrite(os.path.join(save_path, 'mouth_mask.png'), c*255)
            cv2.imwrite(os.path.join('/mnt/d/data/smile/show', f'{case}.png'), d)
            cv2.imwrite(os.path.join(save_path, 'edge.png'), e*255)
            cv2.imwrite(os.path.join(save_path, 'up_edge.png'), f*255)
            # break  
        except:
            print(case)

def test_single():
    smile = predictor()
    path = '/mnt/d/data/smile/Teeth_simulation_10K/'

    case = 'C01002721271'
    # case = '0b01e19ba8f3daa5aed0653dd253a78e'
    img_folder = os.path.join(path,case)
    # if not os.path.exists(os.path.join(img_folder, '微笑像.jpg')) or os.path.exists(os.path.join(img_folder, 'modal', 'blend.png')):
    #     continue
    try:
        face_img = np.array(Image.open(os.path.join(img_folder, '微笑像.jpg')))
    except:
        print('img', case)
    tooth_dict = {int(os.path.basename(p).split('/')[-1][:2]): trimesh.load(p) for p in glob.glob(os.path.join(img_folder, 'models', '*._Root.stl'))}
    steps_dict = {}
    step_one_dict = {}
    
    step_list = natsorted(glob.glob(img_folder+'/models/*.txt'))
    for arr in np.loadtxt(step_list[0]):
        trans = np.eye(4,4)
        trans[:3,3] = arr[1:4]
        trans[:3,:3] = R.from_quat(arr[-4:]).as_matrix()
        step_one_dict[str(int(arr[0]))] = trans
    steps_dict['step_0'] = step_one_dict
    # try:
    a,b,c,d = smile.predict(face_img, tooth_dict, steps_dict)
    # save_path = os.path.join(img_folder, 'modal')
    # os.makedirs(save_path, exist_ok=True)
    # cv2.imwrite(os.path.join(save_path, 'teeth_3d.png'), a)
    # cv2.imwrite(os.path.join(save_path, 'mouth.png'), b)
    # cv2.imwrite(os.path.join(save_path, 'mouth_mask.png'), c*255)
    # cv2.imwrite(os.path.join(save_path, 'blend.png'), d)
if __name__=="__main__":
    test()    