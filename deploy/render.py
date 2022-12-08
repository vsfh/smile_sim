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

camera_dict= {
    'z_init':400,
    'z_init_width':33,
    'z_change':7/90,
    'y_init_11_low':144,
    'y_change':4,
    'y_init_31_up':137
}

def mask_width(mask):
    bg = 1e6
    lin = np.concatenate([x * np.ones((256,1)) for x in np.linspace(0, 255, 256)], 1)
    lin_mask = lin * mask
    lower_mask = np.max(lin_mask, 1)

    lin_mask[lin_mask==0]=bg
    upper_mask = np.min(lin_mask, 1)
    upper_mask[upper_mask==bg] = 0
    return upper_mask, lower_mask

def mask_length(mask):
    bg = 1e6
    lin = np.concatenate([x * np.ones((1,256)) for x in np.linspace(0, 255, 256)], 0)
    lin_mask = lin * mask
    lower_mask = np.max(lin_mask, 0)

    lin_mask[lin_mask==0]=bg
    upper_mask = np.min(lin_mask, 0)
    upper_mask[upper_mask==bg] = 0
    return upper_mask, lower_mask

def _edge_pred(tid, edgeu, edged, mask):
    threshold = [3, None, None]
    edgeu_up, edgeu_down = mask_length(edgeu)
    edged_up, edged_down = mask_length(edged)
    mask_11 = np.zeros_like(tid)
    mask_11[tid==11]=1

    if np.sum(mask_11)<10:
        mask_21 = np.zeros_like(tid)
        mask_21[tid==21]=1
        if np.sum(mask_21) < 10:
            return 0,0,470,0
        else:
            teeth_mask = mask_21
    else:
        teeth_mask = mask_11


    left_11, right_11 = mask_width(teeth_mask)
    up_11, down_11 = mask_length(teeth_mask)

    pos_left_11 = left_11[left_11>0]
    pos_right_11 = right_11[right_11>0]
    width_11 = np.mean(pos_right_11[int(len(pos_right_11)/3):int(len(pos_right_11)/3*2)])\
               -np.mean(pos_left_11[int(len(pos_left_11)/3):int(len(pos_left_11)/3*2)])
    # width_11 = np.max(right_11-left_11)
    width_11 = width_11.clip(0,35)

    camera_z = camera_dict['z_init']-(width_11-camera_dict['z_init_width']-3)/camera_dict['z_change']
    camera_y = (np.mean(down_11[down_11>0])-camera_dict['y_init_11_low']-(width_11-camera_dict['z_init_width'])/2)/camera_dict['y_change']
    if np.sum(mask_11)<10:
        camera_x = ((np.max(left_11))-127)/camera_dict['y_change']
    else:
        camera_x = ((np.max(right_11))-127)/camera_dict['y_change']

    camera_x = camera_x.clip(-2.5,2.5)
    pos_edgeu_down = edgeu_down[edgeu_down>0]
    pos_edged_up = edged_up[edged_up>0]
    dist = np.mean(pos_edged_up[int(len(pos_edged_up)/3):int(len(pos_edged_up)/3*2)])\
               -np.mean(pos_edgeu_down[int(len(pos_edgeu_down)/3):int(len(pos_edgeu_down)/3*2)])

    if dist>threshold[0]:
        dist = dist+camera_dict['y_init_11_low']-camera_dict['y_init_31_up']
        dist_lower = dist/camera_dict['y_change']
    else:
        dist_lower = 0
        
    zero_ = 0
    dist_up = torch.tensor([zero_,zero_,zero_]).unsqueeze(0)
    dist_down = torch.tensor([zero_,zero_,dist_lower]).unsqueeze(0)
    angle = torch.tensor([zero_-1.396,zero_,zero_]).unsqueeze(0)
    movement = torch.tensor([-camera_x, -camera_y, camera_z]).unsqueeze(0)

    renderer = render_init()
    upper, lower, mid = load_up_low(file_path, show=False)
    pred_edge = render(renderer=renderer, upper=upper, lower=lower, mask=mask, angle=angle, movement=movement, dist=[dist_up, dist_down], mid=mid)
    pred_edge = cv2.dilate(pred_edge, np.ones((3,3)))        
    return pred_edge

def render(renderer, upper, lower, mask, angle, movement, dist):

    R = axis_angle_to_matrix(angle)
    T = movement
    edge_align = draw_edge(upper, lower, renderer, dist, mask, R, T)
    return edge_align

def draw_edge(upper, lower, renderer, dist, mask, R, T):
    upper = meshes_to_tensor(upper)
    lower = meshes_to_tensor(lower)

    teeth_batch = join_meshes_as_batch([upper, lower.offset_verts(dist[1][0])])
    deepmap, depth = renderer(meshes_world=teeth_batch, R=R, T=T)

    deepmap = deepmap.detach().cpu().numpy()
    
    teeth_gray = deepmap.astype(np.uint8)
    # cv2.waitKey(0)

    up_edge, low_edge, all_edge = deepmap_to_edgemap(teeth_gray, mask, show=False)
    return all_edge


def deepmap_to_edgemap(teeth_gray, mid, mask, show=False):
    # teeth_gray[teeth_gray == mid+1] = 0
    # teeth_gray[teeth_gray == teeth_gray.max()] = 0
    
    teeth_gray = teeth_gray*mask.detach().numpy()
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

        bg = torch.ones((1, H, W), device=zbuf.device) * (bg_value - 1)
        zbuf_with_bg = torch.cat([bg, zbuf[..., 0]], dim=0)
        teeth = torch.argmin(zbuf_with_bg, dim=0)

        return teeth, 1
    

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