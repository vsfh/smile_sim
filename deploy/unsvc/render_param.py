import trimesh
import os 
import numpy as np
from scipy.spatial.transform import Rotation as R
from glob import glob
import smile_utils
from pytorch3d.renderer import (
	RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
	PerspectiveCameras, SoftPhongShader,HardPhongShader, TexturesVertex, PointLights,SoftSilhouetteShader
)
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
import cv2
import torch
import torch.nn as nn
from pytorch3d.transforms.rotation_conversions import *
from natsort import natsorted

class DepthShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None, light=None, zfar=200, znear=1):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.light = light if blend_params is not None else PointLights()
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
                
        zbuf = fragments.zbuf[...,0]
        zbuf_ = zbuf.clone()
        zbuf_[zbuf_==-1] = 1e10
        zbuf_ = torch.cat((torch.ones_like(zbuf_[0][None])*1e10,zbuf_),0)
        zbuf_mask = torch.argmin(zbuf_, 0, keepdim=True)
        
        for i in range(len(prob_map)):
            prob_map[i] = zbuf[i]*(zbuf_mask[0]==i+1)
        prob_map = torch.sum(prob_map, 0)
        
        out_im = 255*(1-(prob_map-prob_map[prob_map>0].min())/(prob_map.max()-prob_map[prob_map>0].min()))
        out_im[prob_map==0] = 0
        out_im = out_im.detach().cpu().numpy().astype(np.uint8)
        
        return out_im, zbuf_mask

class EdgeShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None, zfar=200, znear=1):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.zfar = zfar
        self.znear = znear

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
  
def deepmap_to_edgemap(teeth_gray, mid):
    teeth_gray = teeth_gray.astype(np.uint8)

    up_teeth = teeth_gray * (teeth_gray <= mid)
    down_teeth = teeth_gray * (teeth_gray > mid)

    kernelx = np.array([[1, -1], [0, 0]])
    kernely = np.array([[1, 0], [-1, 0]])

    gradx = cv2.filter2D(up_teeth, cv2.CV_32F, kernelx)
    grady = cv2.filter2D(up_teeth, cv2.CV_32F, kernely)
    grad = np.abs(gradx) + np.abs(grady)
    up_edge = (grad > 0).astype(np.uint8) * 255

    gradx = cv2.filter2D(down_teeth, cv2.CV_32F, kernelx)
    grady = cv2.filter2D(down_teeth, cv2.CV_32F, kernely)
    grad = np.abs(gradx) + np.abs(grady)
    down_edge = (grad > 0).astype(np.uint8) * 255

    return up_edge, down_edge
  
def get_target_teeth(img_folder, target_step, type='batch', half=False):
    tooth_dict = smile_utils.load_teeth({int(os.path.basename(p).split('/')[-1][:2]): trimesh.load(p) for p in glob(os.path.join(img_folder, 'models', '*._Root.stl'))},half=half)
    step_one_dict = {}
    
    for arr in np.loadtxt(os.path.join(img_folder, target_step)):
        trans = np.eye(4,4)
        trans[:3,3] = arr[1:4]
        trans[:3,:3] = R.from_quat(arr[-4:]).as_matrix()
        step_one_dict[str(int(arr[0]))] = trans
        
    up_mesh = smile_utils.apply_step(tooth_dict, step_one_dict, mode='up', add=False, num_teeth=6)
    up_tensor = smile_utils.meshes_to_tensor(up_mesh,type, device='cuda')
    down_mesh = smile_utils.apply_step(tooth_dict, step_one_dict, mode='down', add=False, num_teeth=6)
    down_tensor = smile_utils.meshes_to_tensor(down_mesh,type, device='cuda')
    return up_tensor, down_tensor

def get_renderer(output_type='EdgeAndDepth', device='cuda', focal_length=12):
    opt_cameras = PerspectiveCameras(device=device, focal_length=focal_length)
    if 'Depth' == output_type:
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=2e-3,
            faces_per_pixel=25,
            perspective_correct=False,
            cull_backfaces=True
        )
        blend_params=BlendParams(sigma=1e-6, gamma=1e-2, background_color=(0., 0., 0.))
        shader = DepthShader(blend_params=blend_params)
    if 'Edge' == output_type:
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=2e-3,
            faces_per_pixel=25,
            perspective_correct=False,
            cull_backfaces=True
        )
        blend_params=BlendParams(sigma=1e-6, gamma=1e-2, background_color=(0., 0., 0.))
        shader = EdgeShader(blend_params=blend_params)
    if 'HardPhong' == output_type:
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=2e-3,
            faces_per_pixel=25,
            perspective_correct=True,
            cull_backfaces=True
        )
        lights = PointLights(device=device, ambient_color=((0.9, 0.9, 0.9),), location=[[2.0, -60.0, -12.0]])
        blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0., 0., 0.))
        shader = HardPhongShader(device=device, cameras=opt_cameras, lights=lights,blend_params=blend_params)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=opt_cameras,
            raster_settings=raster_settings
        ),
        shader=shader
    )
    return renderer
   
def interface(case):
    teeth_mask = cv2.imread(f'/mnt/d/data/smile/out/{case}/teeth_mask.png')
    edge = cv2.imread(f'/mnt/d/data/smile/out/{case}/edge.png')
    
    best_params = torch.load(f'/mnt/d/data/smile/out/{case}/para.pt')
    up_tensor, down_tensor = get_target_teeth(f'/mnt/d/data/smile/Teeth_simulation_10K/{case}','step1.txt')
    renderer = get_renderer('HardPhong', focal_length=best_params['focal_length'])
    T = best_params['T']
    dist = best_params['dist']

    d = 100
    a = 97
    w = 119
    s = 115
    j = 106
    k = 107
    n = 110
    m = 109
    u=117
    i=105
    enter = 13
    angle = 0
    while True:
        with torch.no_grad():
            axis_angles = torch.cat([torch.tensor([0], dtype=torch.float32), torch.tensor([angle], dtype=torch.float32), torch.tensor([0], dtype=torch.float32)],0).cuda()
            R_ = axis_angle_to_matrix(axis_angles[None, :])
            R = R_@best_params['R']
            
            teeth_mesh = join_meshes_as_scene([up_tensor, down_tensor.offset_verts(dist)],
                                                include_textures=True)

            out_im = renderer(meshes_world=teeth_mesh, R=R, T=T)
            out_im = (edge/2+(255/2*out_im[0,:,:,:3]/out_im[0,:,:,:3].max()).detach().cpu().numpy()).astype(np.uint8)
            # out_im[teeth_mask!=0] = teeth_mask
            cv2.imshow('img', out_im)
            key = cv2.waitKey()
            if key==a:
                T[0,0] += 1
            if key==d:
                T[0,0] -= 1
            if key==w:
                T[0,1] += 1
            if key==s:
                T[0,1] -= 1
            if key==j:
                T[0,2] += 5
            if key==k:
                T[0,2] -= 5
            if key==n:
                angle += 3/180
            if key==m:
                angle -= 3/180
            if key==u:
                dist[1] += 2
            if key==i:
                dist[1] -= 2
            if key==enter:
                best_params['T'] = T
                best_params['R'] = R
                best_params['dist'] = dist
                break
    torch.save(best_params, f'/mnt/d/data/smile/out/{case}/para.pt')
    return out_im

def render_depth_mask(case, step_idx=-1, show=False):
    mouth_mask = cv2.imread(f'/mnt/d/data/smile/out/{case}/mouth_mask.png')
    
    best_params = torch.load(f'/mnt/d/data/smile/out/{case}/para.pt')
    step = [file for file in natsorted(os.listdir(f'/mnt/d/data/smile/Teeth_simulation_10K/{case}')) if file.endswith('txt')][step_idx]
    os.makedirs(f'/mnt/d/data/smile/out/{case}/step', exist_ok=True)
    up_tensor, down_tensor = get_target_teeth(f'/mnt/d/data/smile/Teeth_simulation_10K/{case}', step, half=False)
    renderer = get_renderer('Depth', focal_length=best_params['focal_length'])
    T = best_params['T']
    T[0,2]+=50
    dist = best_params['dist']
    R = best_params['R']
    with torch.no_grad():        
        teeth_mesh = join_meshes_as_scene([up_tensor, down_tensor.offset_verts(dist)],
                                            include_textures=True)

        out_im, _ = renderer(meshes_world=teeth_mesh, R=R, T=T)
        
        depth = np.where(mouth_mask[...,0]==0, 0, out_im)
        cv2.imwrite(f'/mnt/d/data/smile/out/{case}/step/depth.png', depth)
        if show:
            cv2.imshow('mat',depth)
            cv2.imshow('mat2',out_im)
            cv2.waitKey(0)
    return
     
def render_edge(case, step_idx=-1, show=False):
    mouth_mask = cv2.imread(f'/mnt/d/data/smile/out/{case}/mouth_mask.png')
    
    best_params = torch.load(f'/mnt/d/data/smile/out/{case}/para.pt')
    step = [file for file in natsorted(os.listdir(f'/mnt/d/data/smile/Teeth_simulation_10K/{case}')) if file.endswith('txt')][step_idx]
    os.makedirs(f'/mnt/d/data/smile/out/{case}/step', exist_ok=True)
    
    up_tensor, down_tensor = get_target_teeth(f'/mnt/d/data/smile/Teeth_simulation_10K/{case}', step, half=False)
    renderer = get_renderer('Edge', focal_length=best_params['focal_length'])
    T = best_params['T']
    T[0,2]+=50
    dist = best_params['dist']
    R = best_params['R']
    with torch.no_grad():        
        teeth_mesh = join_meshes_as_batch([up_tensor, down_tensor.offset_verts(dist)],
                                            include_textures=True)

        out_im, _ = renderer(meshes_world=teeth_mesh, R=R, T=T, extra_mask=None)

        out_im[mouth_mask[...,0]==0]=0  
        up_edge, down_edge = deepmap_to_edgemap(out_im.detach().cpu().numpy(), up_tensor._N)
        cv2.imwrite(f'/mnt/d/data/smile/out/{case}/step/up_edge.png', up_edge)
        cv2.imwrite(f'/mnt/d/data/smile/out/{case}/step/down_edge.png', down_edge)
        
        if show:
            cv2.imshow('mat', up_edge)
            cv2.waitKey(0)
            
    return   

def render_3d(case, step=-1, show=False):
    best_params = torch.load(f'/mnt/d/data/smile/out/{case}/para.pt')
    step_idx = [file for file in natsorted(os.listdir(f'/mnt/d/data/smile/Teeth_simulation_10K/{case}')) if file.endswith('txt')][step]
    os.makedirs(f'/mnt/d/data/smile/out/{case}/step', exist_ok=True)
    up_tensor, down_tensor = get_target_teeth(f'/mnt/d/data/smile/Teeth_simulation_10K/{case}', step_idx, type='scene', half=False)
    renderer = get_renderer('HardPhong', focal_length=best_params['focal_length'])
    T = best_params['T']
    T[0,2]+=50
    dist = best_params['dist']
    R = best_params['R']
    with torch.no_grad():        
        teeth_mesh = join_meshes_as_scene([up_tensor, down_tensor.offset_verts(dist)],
                                            include_textures=True)

        out_im = renderer(meshes_world=teeth_mesh, R=R, T=T, extra_mask=None)
        out_im = (255*out_im[0,:,:,:3]/out_im[0,:,:,:3].max()).detach().cpu().numpy().astype(np.uint8)
        if show:
            cv2.imshow('mat', out_im)
            cv2.waitKey(0)
        cv2.imwrite(f'/mnt/d/data/smile/out/{case}/step/3d.png', out_im)
    return  
    
if __name__=='__main__':
    # interface('C01002745615')
    for case in natsorted(os.listdir('/mnt/d/data/smile/out/'))[-15:]:
        # case = 'C01002757966'
        print(case)
        render_depth_mask(case)
        render_edge(case)
        render_3d(case)
        # break
