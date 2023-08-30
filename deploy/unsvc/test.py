import numpy as np
import torch
import os
import glob
import trimesh
from scipy.spatial.transform import Rotation as R
import cv2
import utils
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
	RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
	PerspectiveCameras, SoftPhongShader,HardPhongShader, TexturesVertex, PointLights,SoftSilhouetteShader
)
Rot = torch.tensor([[[ 9.9987e-01,  1.5944e-02,  3.4199e-04],
         [-2.9583e-03,  1.6436e-01,  9.8640e-01],
         [ 1.5671e-02, -9.8627e-01,  1.6439e-01]]], device='cuda:0')
T = torch.tensor([[ -1.7407,   5.1457, 475.0512]], device='cuda:0')
device = 'cuda:0'

img_folder = '/mnt/e/data/smile/to_b/20220713_SmileyTest_200case/e38be3672450e0155bafbc290b11d0c7'
tooth_dict = {int(os.path.basename(p).split('.')[0][-2:]): trimesh.load(p) for p in glob.glob(os.path.join(img_folder, 'info', 'tooth*.stl'))}
teeth = utils.load_teeth(tooth_dict, type='tooth', half=False, sample=False, voxel_size=1.0)
step = {}
for arr in np.loadtxt('/mnt/e/data/smile/to_b/20220713_SmileyTest_200case/e38be3672450e0155bafbc290b11d0c7/info/step1.txt'):
    trans = np.eye(4,4)
    trans[:3,3] = arr[-3:]
    trans[:3,:3] = R.from_quat(arr[1:5]).as_matrix()
    step[str(int(arr[0]))] = trans
up_mesh = utils.apply_step(teeth, step, mode='up', add=False, num_teeth=6)
up_tensor = utils.meshes_to_tensor(up_mesh, device=device)
down_mesh = utils.apply_step(teeth, step, mode='down', add=False, num_teeth=6)
down_tensor = utils.meshes_to_tensor(down_mesh, device=device)
    
opt_cameras = PerspectiveCameras(device=device, focal_length=torch.tensor(12.9807, device='cuda:0'))
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
teeth_mesh = join_meshes_as_scene([up_tensor, down_tensor],
                                    include_textures=True)
# teeth_mesh.textures = TexturesVertex(verts_features=torch.full([teeth_mesh._N, teeth_mesh._V, 3], 1., device=device))
# deepmap, depth = edge_renderer(meshes_world=teeth_mesh, R=R, T=T)
out_im = edge_renderer(meshes_world=teeth_mesh, R=Rot, T=T)

cv2.imshow('img', (255/(out_im[0,:,:,:3].max())*out_im[0,:,:,:3]*out_im[0,:,:,3][...,None]).detach().cpu().numpy().astype(np.uint8))
cv2.waitKey(0)