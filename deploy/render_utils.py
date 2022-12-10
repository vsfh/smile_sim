import numpy as np
import cv2
import torch
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer
)
from pytorch3d.transforms import axis_angle_to_matrix
import torch.nn as nn
import trimesh
import os
from scipy.spatial.transform import Rotation as R


def draw_edge(upper, lower, renderer, dist, mask, R, T, mid):
    upper = meshes_to_tensor(upper)
    lower = meshes_to_tensor(lower)
    print(mid)

    teeth_batch = join_meshes_as_batch([upper, lower.offset_verts(dist[1][0])])
    deepmap, depth = renderer(meshes_world=teeth_batch, R=R, T=T)

    deepmap = deepmap.detach().numpy()

    teeth_gray = deepmap.astype(np.uint8)

    up_edge, low_edge, all_edge = deepmap_to_edgemap(teeth_gray, mid, mask, show=False)
    return all_edge


def deepmap_to_edgemap(teeth_gray, mid, mask, show=False):
    teeth_gray[teeth_gray == mid+1] = 0
    teeth_gray[teeth_gray == teeth_gray.max()] = 0

    teeth_gray = teeth_gray*mask
    # teeth_gray = teeth_gray[0][0]

    color = set(teeth_gray.flatten())
    for c in color:
        mask = teeth_gray == c


        if np.sum(mask) < 50:
            teeth_gray[mask] = 0
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
        
def load_up_low(teeth_folder_path, show=True):
    num_teeth = 5
    up_keys = list(range(11, 11 + num_teeth)) + list(range(21, 21 + num_teeth))
    down_keys = list(range(31, 31 + num_teeth)) + list(range(41, 41 + num_teeth))
    mid = 0
    up_mesh_list = []
    down_mesh_list = []

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
        mesh = trimesh.load_mesh(mesh_path)
        mesh.apply_transform(M)
        if tid in up_keys:
            mid += 1
            up_mesh_list.append(mesh)
        elif tid in down_keys:
            down_mesh_list.append(mesh)

    upper = trimesh.load_mesh(os.path.join(teeth_folder_path, 'up', 'gum.ply'))
    lower = trimesh.load_mesh(os.path.join(teeth_folder_path, 'down', 'gum.ply'))
    up_mesh_list.append(upper)
    down_mesh_list.append(lower)
    mesh_list = []
    mesh_list += up_mesh_list
    mesh_list += down_mesh_list


    if show:
        trimesh.Scene(mesh_list).show()

    return up_mesh_list, down_mesh_list, mid

def render_init():
    raster_settings = RasterizationSettings(
        image_size=256,
        faces_per_pixel=10,
        perspective_correct=True,
        cull_backfaces=True
    )
    cameras = PerspectiveCameras(device='cpu', focal_length=14)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=EdgeShader())
    return renderer



class EdgeShader(nn.Module):
    def forward(self, fragments, meshes, **kwargs):
        bg_value = 1e6
        zbuf = fragments.zbuf
        N, H, W, K = zbuf.shape
        zbuf[zbuf == -1] = bg_value
        bg = torch.ones((1, H, W), device=zbuf.device) * (bg_value - 1)
        zbuf_with_bg = torch.cat([bg, zbuf[..., 0]], dim=0)
        teeth = torch.argmin(zbuf_with_bg, dim=0)

        return teeth, 1

class renderer:
    def __init__(self) -> None:
        self.model = render_init()
        standard_path = os.path.join(os.path.dirname(__file__), 'standard')
        print(standard_path)
        upper, lower, mid = load_up_low(standard_path, show=False)
        self.upper = upper
        self.lower = lower
        self.mid = mid
    def para_edge(self, mask, angle, movement, dist):
        R = axis_angle_to_matrix(angle)
        T = movement
        edge_align = draw_edge(self.upper, self.lower, self.model, dist, mask, R, T, self.mid)
        return edge_align

render = renderer()