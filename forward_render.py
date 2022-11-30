import os
import torch
import cv2
import torch.nn as nn
import numpy as np
import trimesh
import stl
from pytorch3d.structures import Meshes, join_meshes_as_scene,join_meshes_as_batch
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
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

        return teeth, new_depth.cpu().numpy()

class SEdgeShader(nn.Module):
    def forward(self, fragments, meshes, **kwargs):
        bg_value = 1e6
        zbuf = fragments.zbuf
        N, H, W, K = zbuf.shape

        zbuf[zbuf == -1] = bg_value

        depth, _ = torch.min(zbuf, dim=-1)
        min_depth = (depth.flatten().reshape(depth.shape[0], -1)).min(-1)[0].unsqueeze(-1).unsqueeze(-1)
        depth[depth==bg_value]=0
        max_depth = (depth.flatten().reshape(depth.shape[0], -1)).max(-1)[0].unsqueeze(-1).unsqueeze(-1)
        depth[depth==0]=bg_value
        

        new_depth = 1 - (depth - min_depth) / (max_depth - min_depth)
        new_depth = torch.clip(new_depth, min=0, max=0.5)*2
        return new_depth.unsqueeze(1)

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

import open3d as o3d
def load_mesh(data_file, id):
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(data_file)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()

    mesh.compute_triangle_normals()
    triangle_normals = np.asarray(mesh.triangle_normals)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    mesh.compute_triangle_normals()
    vertices_center = (vertices[triangles[:, 0], 1] + vertices[triangles[:, 1], 1] + vertices[
        triangles[:, 2], 1]) / 3
    if id // 10 in [1, 3]:
        mask = (vertices_center <= 0)
    else:
        mask = (vertices_center >= 0)

    mesh.triangles = o3d.utility.Vector3iVector(
        triangles[mask]
    )
    mesh.triangle_normals = o3d.utility.Vector3dVector(
        triangle_normals[mask]
    )
    mesh = mesh.simplify_vertex_clustering(voxel_size=1.0)
    mesh.compute_triangle_normals()
    return mesh

import copy
def load_up_low(teeth_folder_path, show=True):
    num_teeth = 5
    up_keys = list(range(11, 11 + num_teeth)) + list(range(21, 21 + num_teeth))
    mid = 0
    up_mesh_list = []
    down_mesh_list = []

    text_file = 'TeethAxis_T2.txt'
    filename = 'crown_tooth'

    for line in np.loadtxt(os.path.join(teeth_folder_path, text_file)):
        tid = int(line[0])

        mesh_path = os.path.join(teeth_folder_path, f'{filename}{tid}.stl')
        # stl.mesh.Mesh.from_file(mesh_path).save(mesh_path)
        mesh = load_mesh(mesh_path, tid)
        T = np.eye(4)
        quaternions = line[1:5]
        quaternions = quaternions[[-1, 0, 1, 2]]
        T[:3, :3] = mesh.get_rotation_matrix_from_quaternion(quaternions)
        T[:3, -1] = line[5:]
        T[-1, -1] = 1
        mesh_t = copy.deepcopy(mesh).transform(T)
        if tid in up_keys:
            mid += 1
            up_mesh_list.append(mesh_t)
        else:
            down_mesh_list.append(mesh_t)

    # upper = trimesh.load_mesh(os.path.join(teeth_folder_path, 'up', 'gum.ply'))
    # lower = trimesh.load_mesh(os.path.join(teeth_folder_path, 'down', 'gum.ply'))
    # up_mesh_list.append(upper)
    # down_mesh_list.append(lower)
    mesh_list = []
    mesh_list += up_mesh_list
    mesh_list += down_mesh_list

    return up_mesh_list, down_mesh_list, mid


def render_(renderer,seg_model, upper, lower, mask, angle=-1.396, movement=None, dist=None):
    if dist is None:
        dist = 1
    if movement is None:
        movement = [0, 0, 450]
    R = axis_angle_to_matrix(angle)
    T = movement
    # edge_ori = draw_edge_(seg_model, upper, lower, renderer, dist, mask, R, T)
    up_edge, down_edge = draw_single_(seg_model, upper, lower, renderer, dist, mask, R, T)
    
    return up_edge, down_edge


def draw_edge_(seg_model, upper, lower, renderer, dist, mask, R, T):
    
    upper = meshes_to_tensor(upper).cuda()
    lower = meshes_to_tensor(lower).cuda()

    # teeth_batch = join_meshes_as_batch([join_meshes_as_scene([upper.offset_verts(dist[0][i]), lower.offset_verts(dist[1][i])]) for i in range(R.shape[0])])
    teeth_batch = join_meshes_as_batch([join_meshes_as_scene([upper, lower.offset_verts(dist[i])]) for i in range(R.shape[0])])
    image = renderer(meshes_world=teeth_batch, R=R, T=T, extra_mask=None)
    image = image[..., 0].unsqueeze(1)*mask
    edge = seg_model(image)
    return edge

def draw_single_(seg_model, upper,lower, renderer, dist, mask, R, T):
    
    upper = meshes_to_tensor(upper).cuda()
    lower = meshes_to_tensor(lower).cuda()

    # teeth_batch = join_meshes_as_batch([join_meshes_as_scene([upper.offset_verts(dist[0][i]), lower.offset_verts(dist[1][i])]) for i in range(R.shape[0])])
    teeth_batch_up = join_meshes_as_batch([join_meshes_as_scene([upper.offset_verts(dist[0][i])]) for i in range(R.shape[0])])
    teeth_batch_down = join_meshes_as_batch([join_meshes_as_scene([lower.offset_verts(dist[1][i])]) for i in range(R.shape[0])])
    
    up_edge = renderer(meshes_world=teeth_batch_up, R=R, T=T, extra_mask=None)*mask
    down_edge = renderer(meshes_world=teeth_batch_down, R=R, T=T, extra_mask=None)*mask

    return up_edge, down_edge

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

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
class PLModel(pl.LightningModule):
    def __init__(self):
        super(PLModel, self).__init__()
        # self.save_hyperparameters()
        kwargs = {
            'pretrained': False,
            'features_only': True,
            'feature_location': '',
            'out_indices': [1, 2, 3, 4]
        }

        self.model = smp.Unet(encoder_name='timm-efficientnet-b0',
                              encoder_depth=2,
                              encoder_weights=None,
                              decoder_channels=[32, 16],
                              in_channels=1,
                              activation='sigmoid',
                              classes=1)

    def forward(self, img):
        return self.model(img)

def seg_model():
    seg = PLModel.load_from_checkpoint('/mnt/share/shenfeihong/code/fittingx/weights/epoch=19-val_iou=0.943529.ckpt')
    for param in seg.parameters():
        param.requires_grad = False
    print('load seg model')
    return seg.cuda()

def render_init():
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=2e-3,
        faces_per_pixel=25,
        perspective_correct=False,
        cull_backfaces=True
    )
    cameras = PerspectiveCameras(device='cuda', focal_length=12)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SEdgeShader())
    return renderer

if __name__ == '__main__':
    # upper_jaw, _,_,_ = load_all_teeth('/home/disk/data/tooth_arrangement/1_1000/C01000965707')
    # load_all_teeth('/home/meta/sfh/data/smile/smile_test/test_03_26/C01004890247/info',show=True)
    seg = seg_model()
    renderer = render_init()
    file_path = '/home/disk/data/tooth_arrangement/1_1000/C01001637142'
    upper, lower, mid = load_up_low(file_path, show=True)
    for i in range(10):
        render_(renderer, seg, upper, lower, mid, angle=-1.396, movement=[0,0,400+i*10], dist=0)
