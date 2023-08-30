import cv2
import numpy as np
import os
import open3d as o3d
import copy
import torch
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer import (
    TexturesVertex,
    TexturesUV
)
# from gum_generation.predict_lib import generate_gum
# from gum.gum_deformation.deformer import DeformDLL
# from gum_generation.test_half_jaw import get_result
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def numpy_to_img(n_arr):
    img_array = n_arr[0].transpose(1, 2, 0)
    img_array = img_array.round() * 255
    img_array = img_array.astype(np.uint8)
    img_array = np.squeeze(img_array, axis=2)
    return img_array


def preprocess_mask(img):
    '''

	:param img: 256*256 numpy array with values in (0,50,100,250)
	:return: 1*4*256*256 numpy array with values in (0,1)
	'''

    MASK_COLORMAP = [0, 50, 100, 250]
    # label_list = ['background', 'mouth', 'edge', 'tooth']
    n_label = len(MASK_COLORMAP)
    img = np.array(img, dtype=np.uint8)
    h, w = img.shape[:2]
    onehot_label = np.zeros((n_label, h, w))
    colormap = np.array(MASK_COLORMAP).reshape(n_label, 1, 1)
    colormap = np.tile(colormap, (1, h, w))
    for idx, color in enumerate(MASK_COLORMAP):
        onehot_label[idx] = colormap[idx] == img
    return onehot_label[np.newaxis, :, :, :].astype(np.float32)


def mask_process(im_r, mask, edge):
    mask[mask != 0] = 255
    edge[edge != 0] = 255
    edge = cv2.dilate(edge.astype('uint8'), np.ones((3, 3), np.uint8))
    mask[im_r == 0] = 0
    im_r[mask == 255] = 250
    edge[im_r == 0] = 0
    im_r[edge == 255] = 100
    return preprocess_mask(im_r)


def complex_imgaug(x, mask):
    '''
	Args:
		x: 256*256*3
		mask: 256*256*1
	Returns: 256*256*3
	'''
    aug_img = cv2.blur(x, (15, 15), cv2.BORDER_DEFAULT)
    aug_img = np.where(mask == 0, x, aug_img)
    aug_img = aug_img.transpose(2, 0, 1)[np.newaxis, :, :, :] / 255. * 2 - 1
    aug_img = aug_img.astype(np.float32)
    return aug_img


def resize(image, side_length):
    height, width = image.shape[:2]
    # ratio = width / height

    new_width, new_height = side_length

    if (new_width / width) < (new_height / height):
        scale = new_width / width
        width = new_width
        height = int(height * scale)
    else:
        scale = new_height / height
        height = new_height
        width = int(width * scale)

    image = cv2.resize(image, (width, height))

    return image, scale


def pad(image, side_length):
    new_width, new_height = side_length

    height, width = image.shape[:2]
    dx = (new_width - width) // 2
    dy = (new_height - height) // 2

    if len(image.shape) == 3:
        image = np.pad(image, ((dy, new_height - height - dy), (dx, new_width - width - dx), (0, 0)), mode='constant')
    else:
        image = np.pad(image, ((dy, new_height - height - dy), (dx, new_width - width - dx)), mode='constant')
    return image, (dx, dy)


def resize_and_pad(image, side_length, kps=None, resize_ratio=1.):
    image, scale = resize(image, [int(i * resize_ratio) for i in side_length])
    image, offsets = pad(image, side_length)
    if kps is not None:
        kps = np.array(kps)
        kps *= scale
        kps += offsets

    meta = {
        'scale': scale,
        'offsets': offsets,
        'kps': kps
    }
    return image, meta


def loose_bbox(coords, image_size, loose_coef=0.):
    w, h = image_size
    coords = coords.copy()
    roi_w, roi_h = coords[2] - coords[0], coords[3] - coords[1]

    if isinstance(loose_coef, float):
        left, top, right, bottom = loose_coef, loose_coef, loose_coef, loose_coef
    else:
        left, top, right, bottom = loose_coef

    coords[0] -= roi_w * left
    coords[1] -= roi_h * top
    coords[2] += roi_w * right
    coords[3] += roi_h * bottom

    coords[0] = max(0, int(coords[0]))
    coords[1] = max(0, int(coords[1]))
    coords[2] = min(w, int(coords[2]))
    coords[3] = min(h, int(coords[3]))
    return coords


def normalize_img(img):
    if img.dtype not in ['float32']:
        img = img.astype(np.float32)

    if img.max() > 10.:
        img = img / 255

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


def load_step_file(step_file):
    step = {int(s[0]): s[1:] for s in step_file}
    return step

def generate_gum_with_deform(teeth_dict, jaw_name):
    jaw_type = (1, 2) if jaw_name == 'Upper' else (3, 4)
    result = generate_gum({fdi: tooth for fdi, tooth in teeth_dict.items() if fdi // 10 in jaw_type}, production=True)

    return DeformDLL(result[0], get_result(result)).get_gum()

def load_gum(gum_dict, sample=True, voxel_size=0.2, export=False):
 
    upper = gum_dict['Upper']
    lower = gum_dict['Lower']
    if export:
        upper.export(f'export/gum_u.stl')
        lower.export(f'export/gum_l.stl')
        
    
    vertices = o3d.utility.Vector3dVector(upper.vertices)
    triangles = o3d.utility.Vector3iVector(upper.faces)
    upper = o3d.geometry.TriangleMesh(vertices, triangles)
    vertices = o3d.utility.Vector3dVector(lower.vertices)
    triangles = o3d.utility.Vector3iVector(lower.faces)
    lower = o3d.geometry.TriangleMesh(vertices, triangles)
    
    mesh_combined = upper + lower
    if sample:
        upper = upper.simplify_vertex_clustering(voxel_size=voxel_size)
        lower = lower.simplify_vertex_clustering(voxel_size=voxel_size)
        
    return upper, lower
    
def load_teeth(teeth_dict, type='tooth', half=True, sample=True, voxel_size=0.4):

    teeth = {}
    for key, tooth in teeth_dict.items():
        id = int(key)

        vertices = o3d.utility.Vector3dVector(tooth.vertices)
        triangles = o3d.utility.Vector3iVector(tooth.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, triangles)
        
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

        teeth[id] = mesh
    return teeth

color = [
	[62, 70, 93], [100, 60, 82], [77, 78, 98], [69, 76, 106],
	[98, 109, 63], [80, 59, 103], [101, 96, 98], [66, 98, 100],
	[106, 117, 103], [101, 81, 61], [62, 119, 113], [112, 112, 79],
	[93, 110, 57], [84, 113, 80], [109, 79, 64], [105, 84, 91],

	[216, 193, 223], [230, 248, 230], [227, 226, 180], [218, 235, 236],
	[202, 192, 241], [219, 160, 175], [192, 214, 239], [188, 167, 239],
	[211, 245, 238], [240, 209, 173], [212, 187, 186], [215, 214, 180],
	[232, 173, 237], [180, 165, 227], [235, 217, 157], [187, 217, 162]
]
up_labels = list(range(11, 19)) + list(range(21, 29))
down_labels = list(range(31, 39)) + list(range(41, 49))
colormap = {id: np.array(color[i], dtype=np.float64) / 255 for i, id in enumerate(up_labels + down_labels)}

import trimesh
def trimesh_load_apply(teeth, step):
    from scipy.spatial.transform import Rotation as R
    meshes = {}
    for id in teeth.keys():
        if id not in teeth:
            continue

        mesh = teeth[id]
        transformation = step[id]

        translate = transformation[:3]
        quaternions = transformation[3:]
        quaternions = quaternions[[-1, 0, 1, 2]]

        T = np.eye(4)
        T[:3, :3] = R.from_quat(quaternions).as_matrix()
        T[:3, -1] = translate
        T[-1, -1] = 1

        mesh_t = copy.deepcopy(mesh).apply_transform(T)
        meshes[id] = mesh_t
    return meshes

def apply_step_dict(teeth, step):
    meshes = {}
    for id in teeth.keys():
        if id not in teeth:
            continue

        mesh = teeth[id]
        mesh.paint_uniform_color(colormap[id])
        transformation = step[id]

        translate = transformation[:3]
        quaternions = transformation[3:]
        quaternions = quaternions[[-1, 0, 1, 2]]

        T = np.eye(4)
        T[:3, :3] = mesh.get_rotation_matrix_from_quaternion(quaternions)
        T[:3, -1] = translate
        T[-1, -1] = 1

        mesh_t = copy.deepcopy(mesh).transform(T)
        vertices = mesh_t.vertices
        faces = mesh_t.triangles
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        meshes[id] = trimesh_mesh
    return meshes

def apply_step(teeth, step, keys=None, mode='all', add=True, num_teeth=None, export=False):
    if keys is None:
        if num_teeth is not None:
            if num_teeth>=10:
                keys = {num_teeth}
            else:
                up_keys = list(range(11, 11 + num_teeth)) + list(range(21, 21 + num_teeth))
                down_keys = list(range(31, 31 + num_teeth)) + list(range(41, 41 + num_teeth))
                if mode == 'up':
                    keys = up_keys
                elif mode == 'down':
                    keys = down_keys
                else:
                    keys = up_keys + down_keys
        else:
            keys = {11,21}

    meshes = []
    for id in keys:
        if id not in teeth:
            continue
        # if str(id) not in step:
        #     print('missing step', id)
        #     continue
        mesh = teeth[id]
        mesh.paint_uniform_color(colormap[id])
        transformation = step[str(id)]

        # if np.linalg.norm(transformation[:3])>2:
        #     translate = transformation[:3]
        #     quaternions = transformation[3:]
        # else:
        #     translate = transformation[4:]
        #     quaternions = transformation[:4]
        # quaternions = quaternions[[-1, 0, 1, 2]]

        # T = np.eye(4)
        # T[:3, :3] = mesh.get_rotation_matrix_from_quaternion(quaternions)
        # T[:3, -1] = translate
        # T[-1, -1] = 1
        T = np.array(transformation)

        mesh_t = copy.deepcopy(mesh).transform(T)
        meshes.append(mesh_t)
        if export:
            trimesh.Trimesh(vertices=mesh_t.vertices, faces=mesh_t.triangles).export(f'export/{id}.stl')
    if add:
        mesh_combined = o3d.geometry.TriangleMesh()
        for m in meshes:
            mesh_combined += m
        return [mesh_combined]
    else:
        return meshes


def meshes_to_tensor(meshes, device='cpu'):
    if not isinstance(meshes, list):
        meshes = [meshes]

    verts = []
    faces = []
    tensors = []
    for m in meshes:
        v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
        verts.append(v)

        f = torch.tensor(np.asarray(m.triangles), dtype=torch.long, device=device)
        faces.append(f)
        
        tensors.append(Meshes(
            verts=[v],
            faces=[f],
            textures=TexturesVertex(verts_features=torch.full([1, v.shape[0], 3], 1.8, device=device))
        ))

    verts_rgb = torch.tensor(np.asarray(meshes[0].vertex_colors), dtype=torch.float32)  # (1, V, 3)
    # mesh_tensor = Meshes(
    #     verts=verts,
    #     faces=faces,
    #     # textures=textures
    # )
    mesh_tensor = join_meshes_as_scene(tensors, include_textures=True)
    # mesh_tensor.textures = TexturesVertex(verts_features=torch.full([len(verts), mesh_tensor._V, 3], 0., device=device))
    return mesh_tensor


def get_step_array(case_dir='/home/meta/sfh/gitee_done/OnnxConvert/test_images/C01004890078'):
    info_dir = os.path.join(case_dir, 'info')
    teeth = load_teeth(info_dir, type='tooth', half=True, sample=True, voxel_size=1.0)
    step_files = [f for f in os.listdir(info_dir) if 'step' in f and 'txt' in f]
    step_files = sorted(step_files, key=lambda f: int(f.split('.')[0][4:]))
    mesh_array = []
    idx_array = []
    for file in step_files:
        step1_file = os.path.join(info_dir, file)
        step1 = load_step_file(step1_file)
        up_mesh = apply_step(teeth, step1, mode='up', num_teeth=5, add=True)
        up_verts, up_faces = meshes_to_tensor(up_mesh)
        down_mesh = apply_step(teeth, step1, mode='down', num_teeth=5, add=True)
        down_verts, down_faces = meshes_to_tensor(down_mesh)
        mesh_array.append(np.concatenate([up_verts, up_faces, down_verts, down_faces], axis=1))
        idx_array.append([up_verts.shape[1], up_faces.shape[1], down_verts.shape[1], down_faces.shape[1]])
    mesh_array = np.concatenate(mesh_array, axis=1)
    idx_array = np.array(idx_array)
    return mesh_array[0], idx_array

def deepmap_to_edgemap(teeth_rgb, mouth_mask, mid):
    teeth_gray = teeth_rgb * mouth_mask
    teeth_gray = teeth_gray.astype(np.uint8)

    # max_value = teeth_gray.max()
    # teeth_gray[teeth_gray==max_value] = 0

    # max_value = teeth_gray.max()
    # teeth_gray[teeth_gray==max_value] = 0
    
    color = set(teeth_gray.flatten())
    for c in color:
        mask = teeth_gray == c

        if np.sum(mask) < 20:
            teeth_gray[mask] = 0.
            continue

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


if __name__ == '__main__':
    get_step_array()
