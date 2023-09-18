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

class EdgeAndDepthShader(nn.Module):
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
        
        out_im = (255*(1-(prob_map-prob_map[prob_map>0].min())/(prob_map.max()-prob_map[prob_map>0].min()))).detach().cpu().numpy().astype(np.uint8)
        
        return out_im, zbuf_mask

def get_target_teeth(img_folder, target_step):
    tooth_dict = smile_utils.load_teeth({int(os.path.basename(p).split('/')[-1][:2]): trimesh.load(p) for p in glob(os.path.join(img_folder, 'models', '*._Root.stl'))})
    step_one_dict = {}
    
    for arr in np.loadtxt(os.path.join(img_folder, target_step)):
        trans = np.eye(4,4)
        trans[:3,3] = arr[1:4]
        trans[:3,:3] = R.from_quat(arr[-4:]).as_matrix()
        step_one_dict[str(int(arr[0]))] = trans
        
    up_mesh = smile_utils.apply_step(tooth_dict, step_one_dict, mode='up', add=False, num_teeth=6)
    up_tensor = smile_utils.meshes_to_tensor(up_mesh,'batch', device='cuda')
    down_mesh = smile_utils.apply_step(tooth_dict, step_one_dict, mode='down', add=False, num_teeth=6)
    down_tensor = smile_utils.meshes_to_tensor(down_mesh,'batch', device='cuda')
    return up_tensor, down_tensor

def get_renderer(output_type='EdgeAndDepth', device='cuda', focal_length=12):
    opt_cameras = PerspectiveCameras(device=device, focal_length=focal_length)
    if 'EdgeAndDepth' == output_type:
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=2e-3,
            faces_per_pixel=25,
            perspective_correct=False,
            cull_backfaces=True
        )
        lights = PointLights(device=device, ambient_color=((0.9, 0.9, 0.9),), location=[[2.0, -60.0, -12.0]])
        blend_params=BlendParams(sigma=1e-6, gamma=1e-2, background_color=(0., 0., 0.))
        shader = EdgeAndDepthShader(blend_params=blend_params)
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
   
def render(case):
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

if __name__=='__main__':
    for case in os.listdir('/mnt/d/data/smile/out/'):
        print(case)
        render(case)
