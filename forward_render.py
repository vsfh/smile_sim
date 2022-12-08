import os
import glob
import torch
import cv2
import torch.nn as nn
import numpy as np
import trimesh
# from teeth_arrangement.landmark_detection import extract_landmarks
# from teeth_arrangement.tooth import GlobalTooth
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
import stl
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
from pytorch3d.transforms import axis_angle_to_matrix


def meshes_to_tensor(meshes, device='cpu'):
    if not isinstance(meshes, list):
        meshes = [meshes]

    verts = []
    faces = []
    for m in meshes:
        v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
        verts.append(v)

        f = torch.tensor(np.asarray(m.faces), dtype=torch.long, device=device)
        faces.append(f)
    mesh_tensor = Meshes(
        verts=verts,
        faces=faces,
        # textures=textures
    )
    return mesh_tensor


class EdgeShader(nn.Module):
    def forward(self, fragments, meshes, **kwargs):
        bg_value = 1e6
        zbuf = fragments.zbuf
        N, H, W, K = zbuf.shape

        zbuf[zbuf == -1] = bg_value

        depth, _ = torch.min(zbuf[:-1], dim=0)
        max_depth, min_depth = depth[depth != bg_value].max(), depth[depth != bg_value].min()
        new_depth = 1 - (depth - min_depth) / (max_depth - min_depth)
        new_depth[depth == bg_value] = 0

        # cv2.imshow('depth', new_depth.cpu().numpy())
        # cv2.waitKey()

        bg = torch.ones((1, H, W), device=zbuf.device) * (bg_value - 1)
        zbuf_with_bg = torch.cat([bg, zbuf[..., 0]], dim=0)
        teeth = torch.argmin(zbuf_with_bg, dim=0)

        return teeth, new_depth.cpu().numpy()

class SEdgeShader(nn.Module):
    def forward(self, fragments, meshes, **kwargs):
        bg_value = 1e6
        zbuf = fragments.zbuf
        N, H, W, K = zbuf.shape

        # zbuf[zbuf == -1] = bg_value

        # depth, _ = torch.min(zbuf, dim=0)

        # min_depth = (depth.flatten().reshape(depth.shape[0], -1)).min(-1)[0].unsqueeze(-1)
        # depth[depth==bg_value]=0
        # max_depth = (depth.flatten().reshape(depth.shape[0], -1)).max(-1)[0].unsqueeze(-1)
        # depth[depth==0]=bg_value

        # new_depth = 1 - (depth - min_depth) / (max_depth - min_depth)

        # bg = torch.ones((1, H, W), device=zbuf.device) * (bg_value - 1)
        # zbuf_with_bg = torch.cat([bg, zbuf[..., 0]], dim=0)
        # teeth = torch.argmin(zbuf_with_bg, dim=0)
        
        
        alpha = torch.clip(zbuf[...,0], min=0, max=1)
        

        return alpha.unsqueeze(1)

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
        alpha = torch.prod((1.0 - prob_map), dim=-1, keepdim=True)
        alpha = 1 - alpha

        return alpha

def load_up_low(teeth_folder_path, mode,num_teeth=1):

    up_keys = list(range(11, 11 + num_teeth)) + list(range(21, 21 + num_teeth))
    down_keys = list(range(31, 31 + num_teeth)) + list(range(41, 41 + num_teeth))

    mid = 0
    up_mesh_list = []
    down_mesh_list = []

    if mode == 'ori':
        text_file = 'TeethAxis_Ori.txt'
    else:
        text_file = 'TeethAxis_T2.txt'
    filename = 'tooth'

    for line in np.loadtxt(os.path.join(teeth_folder_path, text_file)):
        tid = int(line[0])
        M = np.zeros(shape=(4, 4))
        M[:3, 3] = line[5:]
        r = R.from_quat(line[1:5])

        M[:3, :3] = r.as_matrix()
        M[3, 3] = 1
        mesh_path = os.path.join(teeth_folder_path, f'{filename}{tid}.stl')
        # stl.mesh.Mesh.from_file(mesh_path).save(mesh_path)
        mesh = trimesh.load_mesh(mesh_path)
        mesh.apply_transform(M)
        if tid in up_keys:
            mid += 1
            up_mesh_list.append(mesh)
        elif tid in down_keys:
            down_mesh_list.append(mesh)

    # upper = trimesh.load_mesh(os.path.join(teeth_folder_path, 'up', 'gum.ply'))
    # lower = trimesh.load_mesh(os.path.join(teeth_folder_path, 'down', 'gum.ply'))
    # up_mesh_list.append(upper)
    # down_mesh_list.append(lower)
    mesh_list = []
    mesh_list += up_mesh_list
    mesh_list += down_mesh_list
    return up_mesh_list, down_mesh_list, mid

def render_(renderer, upper, lower, mid, mask, angle=-1.396, movement=None, dist=None, mode='train'):
    R = axis_angle_to_matrix(angle)
    T = movement
    edge_ori = draw_edge_(upper, lower, renderer, dist, mid,mask, R, T, mode)
    return edge_ori


def draw_edge_(upper, lower, renderer, dist, mid,mask, R, T, mode):
    if mode=='train':
        teeth_mesh_lower = join_meshes_as_batch([join_meshes_as_scene([lower.offset_verts(dist[1][i])]) for i in range(R.shape[0])])
        teeth_mesh_upper = join_meshes_as_batch([join_meshes_as_scene([upper.offset_verts(dist[0][i])]) for i in range(R.shape[0])])
    
        deepmap_upper = renderer(meshes_world=teeth_mesh_upper, R=R, T=T)*mask
        deepmap_lower = renderer(meshes_world=teeth_mesh_lower, R=R, T=T)*(mask-deepmap_upper)
        return deepmap_upper, deepmap_lower
    else:
        teeth_mesh = join_meshes_as_batch([join_meshes_as_batch([upper.offset_verts(dist[0][i]), \
                                                                lower.offset_verts(dist[1][i])]) for i in range(R.shape[0])])
        deepmap, _ = renderer(meshes_world=teeth_mesh, R=R, T=T)
        # print(deepmap.shape, mask.squeeze().shape)

        deepmap = deepmap*(mask.squeeze())

        return deepmap

def deepmapToedge(deepmap, mid):
    teeth_gray = deepmap.detach().cpu().numpy().astype(np.uint8)
    # teeth_gray[teeth_gray == mid+1] = 0
    # teeth_gray[teeth_gray == teeth_gray.max()] = 0


    color = set(teeth_gray.flatten())
    for c in color:
        mask = teeth_gray == c
        if np.sum(mask) < 50:
            teeth_gray[mask] = 0.
            continue

    up_teeth = teeth_gray * (teeth_gray > mid)
    down_teeth = teeth_gray * (teeth_gray <= mid)

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
    return up_edge+down_edge   

def deepmap_to_edgemap_(teeth_gray, mouth_mask, mid, show=False):
    teeth_gray[teeth_gray == mid+1] = 0
    teeth_gray[teeth_gray == teeth_gray.max()] = 0

    teeth_gray[mouth_mask==0]=0

    color = set(teeth_gray.flatten())
    for c in color:
        mask = teeth_gray == c
        if np.sum(mask) < 50:
            teeth_gray[mask] = 0.
            continue

    up_teeth = teeth_gray * (teeth_gray > mid)
    down_teeth = teeth_gray * (teeth_gray <= mid)

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



    if show:
        cv2.imshow('img1', up_edge + down_edge)
        cv2.waitKey(0)
    return up_edge, down_edge, up_edge + down_edge


def draw_edge(teeth_mesh, renderer, mask, mid, R, T):
    teeth_mesh = join_meshes_as_batch([meshes_to_tensor(teeth_mesh).cuda()])
    deepmap, depth = renderer(meshes_world=teeth_mesh, R=R, T=T)
    deepmap = deepmap.detach().cpu().numpy()
    teeth_gray = deepmap.astype(np.uint8)

    up_edge, low_edge, all_edge = deepmap_to_edgemap(teeth_gray, mask, mid, show=False)
    return all_edge


def deepmap_to_edgemap(teeth_gray, mouth_mask, mid, show=False):
    max_value = teeth_gray.max()
    teeth_gray[teeth_gray == max_value] = 0
    max_value = teeth_gray.max()
    teeth_gray[teeth_gray == max_value] = 0

    # teeth_gray[mouth_mask==0]=0

    color = set(teeth_gray.flatten())
    for c in color:
        mask = teeth_gray == c

        if np.sum(mask) < 50:
            teeth_gray[mask] = 0.
            continue

    up_teeth = teeth_gray * (teeth_gray > mid)
    down_teeth = teeth_gray * (teeth_gray <= mid)

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
    if show:
        cv2.imshow('img1', up_edge + down_edge)
        # cv2.imshow('img', mouth_mask)
        cv2.waitKey(0)
    return up_edge, down_edge, up_edge + down_edge

def render_init(mode='train'):
    if mode=='train':
        raster_settings = RasterizationSettings(
            image_size=256,
            faces_per_pixel=10,
            perspective_correct=True,
            cull_backfaces=True
        )
        cameras = PerspectiveCameras(device='cuda', focal_length=12)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SEdgeShader())
    else:
        raster_settings = RasterizationSettings(
            image_size=256,
            faces_per_pixel=10,
            perspective_correct=True,
            cull_backfaces=True
        )
        cameras = PerspectiveCameras(device='cuda', focal_length=12)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=EdgeShader())        
    return renderer
from natsort import natsorted
def mask_filter():
    file = open('big_mask.txt', 'w')
    mask_path = '/mnt/share/shenfeihong/data/smile_/seg_6000/'
    ras = np.concatenate([np.ones((1, 256)) * (i + 1) for i in range(256)], axis=0)
    for m_file in os.listdir(mask_path):
        m_path = os.path.join(mask_path, m_file,'MouthMask.png')
        mask = cv2.imread(m_path)
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        mask = (mask / 255).astype(np.uint8)

        res = (ras*mask).reshape(1,-1)[0]
        right = max(res)
        res[res==0]=right
        left = min(res)
        height = right-left
        if height>80:
            print(m_file)
            file.write(m_path+'\n')

    file.close()

def img_gen():
    with open('big_mask.txt','r') as f:
        a = f.readlines()
    renderer = render_init()

    path = '/home/disk/data/tooth_arrangement/1_1000/'
    save = '/mnt/share/shenfeihong/data/smile_/edge'
    for file in os.listdir(path)[:200]:
        print(file)
        #
        file_path = os.path.join(path, file)
        save_path = os.path.join(save, file)
        i = np.random.randint(len(a) - 1)
        mask_file = a[i]
        print(mask_file.replace('\n',''))
        mask = cv2.imread(mask_file.replace('\n',''))
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        mask = (mask / 255).astype(np.uint8)

        dist = [np.random.randint(-6,-3),np.random.randint(3,6)]
        movement = [0, 0, 470+np.random.randint(-3,3)*10]

        # dist = [0,0]
        # movement = [0,0,475]
        upper_ori, lower_ori, mid_ori = load_up_low(file_path, mode='ori', show=False)
        edge_ori = render_(renderer,upper_ori, lower_ori, mid_ori,mask, movement=movement, dist=dist)
        upper, lower, mid = load_up_low(file_path, mode='T2', show=False)
        edge_align = render_(renderer,upper, lower, mid, mask, movement=movement, dist=dist)

        # cv2.imshow('align', edge_align)
        # cv2.imshow('align1', edge_ori)
        # cv2.waitKey(0)
        # continue

        os.makedirs(save_path,exist_ok=True)
        cv2.imwrite(os.path.join(save_path, 'ori.png'), edge_ori)
        cv2.imwrite(os.path.join(save_path, 'align.png'), edge_align)
        cv2.imwrite(os.path.join(save_path, 'mask.png'), mask.astype(np.uint8) * 255)

def img_gen_single():

    renderer = render_init()

    file_path = '/mnt/share/shenfeihong/data/smile_/C01001659595'
    movement = [0, 0, 470]
    # upper_ori, lower_ori, mid_ori = load_up_low(file_path, mode='ori', show=False)
    # edge_ori = render_(renderer,upper_ori, lower_ori, mid_ori,1, movement=movement)
    upper, lower, mid = load_up_low(file_path, mode='T2', show=False)
    edge_align = render_(renderer,upper, lower, mid, 1, movement=movement)

