import os
import glob
import torch
import cv2
import torch.nn as nn
import numpy as np
import trimesh
# from teeth_arrangement.landmark_detection import extract_landmarks
# from teeth_arrangement.tooth import GlobalTooth
# from natsort import natsorted
from scipy.spatial.transform import Rotation as R
# import stl
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
from pytorch3d.transforms import axis_angle_to_matrix
import open3d as o3d
import stl
def load_single_teeth_mesh(data_file, id=0,half=True, sample=True, voxel_size=1.0):
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(data_file)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()

    mesh.compute_triangle_normals()
    triangle_normals = np.asarray(mesh.triangle_normals)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    mesh.compute_triangle_normals()
    if half:
        vertices_center = (vertices[triangles[:, 0], 1] + vertices[triangles[:, 1], 1] + vertices[
            triangles[:, 2], 1]) / 3
        if id // 10 in [1, 3]:
            mask = (vertices_center <= 0)
        # mask = (vertices_center <= 0) & (rot_y <= 0)
        else:
            mask = (vertices_center >= 0)
        # mask = (vertices_center >= 0) & (rot_y <= 0)

        mesh.triangles = o3d.utility.Vector3iVector(
            triangles[mask]
        )
        mesh.triangle_normals = o3d.utility.Vector3dVector(
            triangle_normals[mask]
        )

    if sample:
        # print(np.asarray(mesh.triangles).shape)
        mesh = mesh.simplify_vertex_clustering(voxel_size=voxel_size)
    # print(np.asarray(mesh.triangles).shape)
    return mesh

def meshes_to_tensor(meshes, device='cpu'):
    if not isinstance(meshes, list):
        meshes = [meshes]

    verts = []
    faces = []
    for m in meshes:
        v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
        verts.append(v)

        f = torch.tensor(np.asarray(m.triangles), dtype=torch.long, device=device)
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

        return teeth, new_depth.cpu().detach().numpy()


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


def load_all_teeth(teeth_folder_path, up_rand=0, down_rand=0, mode='ori', show=False, offset=True):
    num_teeth = 5
    up_keys = list(range(11, 11 + num_teeth)) + list(range(21, 21 + num_teeth))
    down_keys = list(range(31, 31 + num_teeth)) + list(range(41, 41 + num_teeth))
    
    mid = 0
    up_mesh_list = []
    down_mesh_list = []
    if mode == 'ori':
        text_file = 'TeethAxis_Ori.txt'
        filename = 'crown_tooth'
    else:
        text_file = 'TeethAxis_T2.txt'
        filename = 'crown_tooth'

    for line in np.loadtxt(os.path.join(teeth_folder_path, text_file)):
        tid = int(line[0])
        M = np.zeros(shape=(4, 4))
        # M[:3,3] = line[1:4]
        # r = R.from_quat(line[4:])
        M[:3, 3] = line[5:]
        if offset:
            if tid in up_keys:
                M[2, 3] = M[2, 3] - up_rand
            elif tid in down_keys:
                M[2, 3] = M[2, 3] + down_rand
        r = R.from_quat(line[1:5])

        M[:3, :3] = r.as_matrix()
        M[3, 3] = 1
        mesh_path = os.path.join(teeth_folder_path, f'{filename}{tid}.stl')
        stl.mesh.Mesh.from_file(mesh_path).save(mesh_path)
        mesh = trimesh.load_mesh(mesh_path)
        mesh.apply_transform(M)
        if tid in up_keys:
            mid += 1
            up_mesh_list.append(mesh)
        elif tid in down_keys:
            print(tid)
            down_mesh_list.append(mesh)
    mesh_list = []
    mesh_list += up_mesh_list
    mesh_list += down_mesh_list
    if mode == 'ori':
        upper = trimesh.load_mesh(os.path.join(teeth_folder_path, 'upper_gum.stl'))
        lower = trimesh.load_mesh(os.path.join(teeth_folder_path, 'lower_gum.stl'))
    else:
        upper = trimesh.load_mesh(os.path.join(teeth_folder_path, 'up', 'gum.ply'))
        lower = trimesh.load_mesh(os.path.join(teeth_folder_path, 'down', 'gum.ply'))
    if offset:
        M = np.eye(4, 4)
        M[2, 3] = -up_rand
        upper.apply_transform(M)
        M[2, 3] = down_rand
        lower.apply_transform(M)
    mesh_list.append(lower)
    mesh_list.append(upper)
    if show:
        trimesh.Scene(mesh_list).show()

    return mesh_list, mid


def render(renderer, upper, lower, mask, angle, movement, dist, mid):

    R = axis_angle_to_matrix(angle).cuda()
    T = movement
    edge_align = draw_edge(upper, lower, renderer, dist, mask, mid, R, T)
    return edge_align

import copy

def load_up_low(teeth_folder_path, num_teeth=5, show=True):
    
    up_keys = list(range(11, 11 + num_teeth)) + list(range(21, 21 + num_teeth))
    down_keys = list(range(31, 31 + num_teeth)) + list(range(41, 41 + num_teeth))
    mid = 0
    up_mesh_list = []
    down_mesh_list = []
    if show:
        show_list = []

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
        mesh_t = load_single_teeth_mesh(mesh_path, tid)
        if show:
            s_mesh = trimesh.load_mesh(mesh_path)
            s_mesh.apply_transform(M)
            show_list.append(s_mesh)
        mesh = copy.deepcopy(mesh_t).transform(M)
        if tid in up_keys:
            mid += 1
            up_mesh_list.append(mesh)
        elif tid in down_keys:
            down_mesh_list.append(mesh)

    upper = load_single_teeth_mesh(os.path.join(teeth_folder_path, 'up', 'gum.ply'), half=False)
    lower = load_single_teeth_mesh(os.path.join(teeth_folder_path, 'down', 'gum.ply'), half=False)
    up_mesh_list.append(upper)
    down_mesh_list.append(lower)
    mesh_list = []
    mesh_list += up_mesh_list
    mesh_list += down_mesh_list


    if show:
        trimesh.Scene(show_list).show()

    return up_mesh_list, down_mesh_list, mid


def render_(renderer, upper, lower, mid, angle=-1.396, movement=None, dist=None):
    if dist is None:
        dist = 1
    if movement is None:
        movement = [0, 0, 450]
    R = axis_angle_to_matrix(torch.tensor([angle, 0, 0], dtype=torch.float32)[None]).cuda()
    T = torch.tensor(movement, dtype=torch.float32)[None].cuda()
    edge_ori = draw_edge_(upper, lower, renderer, dist, mid, R, T)


def draw_edge_(upper, lower, renderer, dist, mid, R, T):
    dist = torch.tensor([0, 0, dist])
    teeth_mesh = join_meshes_as_batch(
        [meshes_to_tensor(upper).cuda(), meshes_to_tensor(lower).offset_verts(dist).cuda()])
    deepmap, depth = renderer(meshes_world=teeth_mesh, R=R, T=T)
    deepmap = deepmap.detach().cpu().numpy()
    teeth_gray = deepmap.astype(np.uint8)

    up_edge, low_edge, all_edge = deepmap_to_edgemap_(teeth_gray, 1, mid, show=True)
    return all_edge
    # cv2.imwrite(os.path.join(save_path, 'up.png'), up_edge)
    # cv2.imwrite(os.path.join(save_path, 'low.png'), low_edge)
    # cv2.imwrite(os.path.join(save_path, 'edge.png'), all_edge)


def deepmap_to_edgemap_(teeth_gray, mouth_mask, mid, show=False):
    teeth_gray[teeth_gray == 15] = 0
    teeth_gray[teeth_gray == 30] = 0

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


def draw_edge(upper, lower, renderer, dist, mask, mid, R, T):
    upper = meshes_to_tensor(upper).cuda()
    lower = meshes_to_tensor(lower).cuda()

    teeth_batch = join_meshes_as_batch([upper, lower.offset_verts(dist[1][0])])
    deepmap, depth = renderer(meshes_world=teeth_batch, R=R, T=T)

    deepmap = deepmap.detach().cpu().numpy()
    
    teeth_gray = deepmap.astype(np.uint8)
    # cv2.waitKey(0)

    up_edge, low_edge, all_edge = deepmap_to_edgemap(teeth_gray, mid, mask, show=False)
    low_edge = cv2.dilate(low_edge, np.ones((3,3)))
    up_edge = cv2.dilate(up_edge, np.ones((3,3)))
    
    return up_edge+low_edge


def deepmap_to_edgemap(teeth_gray, mid, mask, show=False):
    teeth_gray[teeth_gray == mid+1] = 0
    teeth_gray[teeth_gray == teeth_gray.max()] = 0
    
    teeth_gray = teeth_gray*mask.detach().cpu().numpy()
    teeth_gray = teeth_gray[0][0]

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

def render_init():
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
if __name__ == '__main__':
    # upper_jaw, _,_,_ = load_all_teeth('/home/disk/data/tooth_arrangement/1_1000/C01000965707')
    # load_all_teeth('/home/meta/sfh/data/smile/smile_test/test_03_26/C01004890247/info',show=True)
    renderer = render_init()
    file_path = '/mnt/share/shenfeihong/data/smile_/C01001637142'
    upper, lower, mid = load_up_low(file_path, show=False)
    zero_ = torch.tensor([0,0]).unsqueeze(0).cuda()
    angle = torch.tensor([0,0]).unsqueeze(0).cuda()
    pred_edge = render(renderer=renderer, upper=upper, lower=lower, mask=1, angle=angle, movement=movement, dist=dist, mid=mid)
