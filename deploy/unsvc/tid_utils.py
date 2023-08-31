import cv2
import numpy as np
import os.path as osp


def compute_meta(src_shape, dst_shape, resize_ratio=1., bbox=None, align_mode='center', keep=False):
    src_width, src_height = src_shape
    dst_width, dst_height = dst_shape

    if bbox is not None:
        x1, y1, x2, y2 = list(map(int, bbox[:4]))
    else:
        x1, y1 = 0, 0
        x2, y2 = src_width, src_height

    crop_width, crop_height = x2 - x1, y2 - y1

    rdst_width, rdst_height = int(dst_width * resize_ratio), int(dst_height * resize_ratio)
    rw = rdst_width / crop_width
    rh = rdst_height / crop_height
    if rw < rh:
        scale = rw
        resize_width = rdst_width
        resize_height = int(crop_height * scale)
    else:
        scale = rh
        resize_height = rdst_height
        resize_width = int(crop_width * scale)
    if keep:
        resize_height = dst_height
        resize_width = dst_width
        x2 = src_width
        y2 = src_height
    if align_mode == 'center':
        dx = (dst_width - resize_width) // 2
        dy = (dst_height - resize_height) // 2
    else:
        dx = 0
        dy = 0

    offsets = (dx, dy, resize_width + dx, resize_height + dy)
    meta = {
        'scale': scale,
        'offsets': offsets,
        'rect': (x1, y1, x2, y2),
        'image_size': (src_width, src_height),
        'resize_shape': (resize_width, resize_height),
        'dst_shape': (dst_width, dst_height)
    }
    return meta


def preprocess_image(image, dst_shape, resize_ratio=1., bbox=None, align_mode='center', pad_val=0, interpolation=None,
                     keep=False):
    height, width = image.shape[:2]
    if isinstance(dst_shape, int):
        dst_shape = (dst_shape, dst_shape)

    meta = compute_meta((width, height), dst_shape, resize_ratio, bbox, align_mode, keep=keep)
    resize_shape = meta['resize_shape']
    x1, y1, x2, y2 = meta['rect']
    dleft, dtop, dright, dbottom = meta['offsets']

    image = image[y1:y2, x1:x2]
    image = cv2.resize(image, resize_shape, interpolation=interpolation)
    if len(image.shape) != 3:
        image = image[..., None]

    dst_width, dst_height = dst_shape
    image = np.pad(image, ((dtop, dst_height - dbottom), (dleft, dst_width - dright), (0, 0)), mode='constant',
                   constant_values=pad_val)

    return image, meta


def cxywh2xyxy(cxywh):
    cx, cy, w, h = cxywh
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return np.array([x1, y1, x2, y2])


def verify_xyxy(xyxy, image_size):
    width, height = image_size
    x1, y1, x2, y2 = xyxy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(x2, width), min(y2, height)
    return np.array([x1, y1, x2, y2])


def xyxy2cxywh(xyxy):
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    return np.array([cx, cy, w, h])


def loose_bbox(coords, image_size, loose_coef=0.):
    w, h = image_size
    coords = np.array(coords)
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
    coords = coords.astype(int)
    return coords


def normalize_image(image, ):
    if image.max() > 1.:
        image = image.astype(np.float32) / 255
    image = np.transpose(image.astype(np.float32), (2, 0, 1))
    image = image[None]
    return image


def blob_image(input_img, mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    input_tensor = (input_img.astype(np.float32) - mean) / std
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = input_tensor[None]
    return input_tensor


def visualize_keypoints(img, kps, show_size=800, kp_names=None):
    if isinstance(kps, list):
        kps = [(x, y) for x, y in kps]
    elif hasattr(kps, 'keypoints'):
        kps = [(kp.x, kp.y) for kp in kps.keypoints]

    scale = show_size / max(img.shape[0], img.shape[1])
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width * scale), int(height * scale)))
    kps = [(int(x * scale), int(y * scale)) for x, y in kps]

    font_size = 0.4
    for i, (x, y) in enumerate(kps):

        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        if kp_names is None:
            # cv2.putText(img, f'{i + 1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255))
            pass
        else:
            cv2.putText(img, kp_names[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0))

    return img


def smooth_line(line, sample_num, w=None, s=None, k=3, is_closed=False):
    from scipy.interpolate import splrep, splev, splprep
    from scipy import interpolate

    # if is_closed is None:
    #     if np.allclose(line[0], line[-1]):
    #         is_closed = True
    #     else:
    #         is_closed = False

    # _, index = np.unique(line, return_index=True, axis=0)
    # line = line[sorted(index)]

    # if is_closed:
    #     line = np.concatenate([line, line[:1]], axis=0)

    # print(line.shape)
    tck, u = splprep(line.T, u=None, w=w, s=s, k=k, per=is_closed)
    u_new = np.linspace(u.min(), u.max(), sample_num)
    x, y = interpolate.splev(u_new, tck, der=0)
    pts = np.stack([x, y]).T
    return pts


def extract_shapes_from_mask(mask, num=1, smooth=False, min_point_number=40, kernel_size=None):
    from scipy.interpolate import splrep, splev, splprep
    from scipy import interpolate

    if mask.max() == 1:
        mask = mask * 255

    mask = mask.astype(np.uint8)

    if kernel_size is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [(c[..., 0].max() - c[..., 0].min()) * (c[..., 1].max() - c[..., 1].min()) for c in contours]
    # lengths = [cv2.arcLength(c, True) for c in contours]

    contours = [contours[i] for i in np.argsort(areas)[::-1]]

    shapes = []
    for i in range(num):
        if i >= len(contours):
            shapes.append(None)
            continue
        cnt = contours[i]
        if smooth:
            diag = np.linalg.norm(mask.shape[:2])
            sample_num = int(cv2.arcLength(cnt, True) / diag * 800 * 0.1)
            sample_num = max(min_point_number, sample_num)

            line = np.array(cnt[:, 0])
            _, index = np.unique(line, return_index=True, axis=0)
            line = line[sorted(index)]
            line = np.concatenate([line, line[:1]], axis=0)

            m = len(line)
            s = m - (2*m) ** 0.5
            tck, u = splprep(line.T, u=None, s=s, k=3, per=1)
            u_new = np.linspace(u.min(), u.max(), sample_num)
            x, y = interpolate.splev(u_new, tck, der=0)
            pts = np.stack([x, y]).T

        else:
            pts = cnt[:, 0]

        shapes.append(pts)

    return shapes


def generate_map_rel(teeth_dict, disease_dict, threshold=0.2, exclude=None):
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity

    if exclude is None:
        exclude_list = []
    else:
        exclude_list = exclude

    for tid in teeth_dict:
        teeth_poly = Polygon(teeth_dict[tid]['points'])
        if explain_validity(teeth_poly) != 'Valid Geometry':
            print('invalid teeth geometry', tid)
            continue

        teeth_area = teeth_poly.area
        sqrt_area = np.sqrt(teeth_area)

        for did in disease_dict:
            if disease_dict[did]['label'] in exclude_list:
                continue

            disease_poly = Polygon(disease_dict[did]['points'])
            if explain_validity(teeth_poly) != 'Valid Geometry':
                print('invalid disease geometry')
                continue

            disease_area = disease_poly.area
            dist = teeth_poly.distance(disease_poly)
            inter = teeth_poly.intersection(disease_poly)

            if dist / sqrt_area < 0.01 and inter.area / disease_area > 0.3:
                teeth_dict[tid]['map'].append(did)
                disease_dict[did]['map'].append(tid)
    return



def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.
    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).astype(np.float32)
    seg_masks = seg_masks.reshape(n_samples, -1).astype(np.float32)
    # inter.
    inter_matrix = np.matmul(seg_masks, seg_masks.transpose(1, 0))
    # union.
    # sum_masks_x = sum_masks.broadcast_to(n_samples, n_samples)
    sum_masks_x = np.broadcast_to(sum_masks, (n_samples, n_samples))
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix))
    iou_matrix = np.triu(iou_matrix, 1)
    # label_specific matrix.
    # cate_labels_x = cate_labels.expand(n_samples, n_samples)
    cate_labels_x = np.broadcast_to(cate_labels, (n_samples, n_samples))
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).astype(np.float32)
    label_matrix = np.triu(label_matrix, 1)

    # IoU compensation
    compensate_iou = (iou_matrix * label_matrix).max(0)
    # compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)
    compensate_iou = np.broadcast_to(compensate_iou, (n_samples, n_samples))
    compensate_iou = compensate_iou.transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = np.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = np.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


def visualize_heatmap(img, heatmap, alpha=0.5):
    if len(heatmap.shape) == 3:
        heatmap = heatmap[0]

    img = img.copy()
    height, width = img.shape[:2]

    heatmap[heatmap < 0] = 0.
    heatmap = (heatmap / (heatmap.max() + 1e-7) * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (width, height))
    seg_map = np.tile(heatmap[..., np.newaxis], (1, 1, 3))
    # seg_map[..., :] = cv2.resize(heatmap, (width, height))

    img[seg_map > 50] = seg_map[seg_map > 50]
    img = img.astype(np.uint8)
    return img


def is_valid_image(path):
    img_suffix = ["jpg", "png", "jpeg", "bmp", "heic"]
    path_suffix = osp.splitext(osp.basename(path))[-1][1:]
    if path_suffix.lower() not in img_suffix:
        return False
    return True


def cal_dist(p, a, b):
    ab = b - a
    ap = p - a
    return np.cross(ab, ap) / np.linalg.norm(ab)


def cal_mid(line, a=None, b=None):
    if a is None:
        a = line[0]

    if b is None:
        b = line[-1]
    a = np.array(a)
    b = np.array(b)

    dists = [cal_dist(p, b, a) for p in line]
    return dists


def rotate_pts(points, angle, origin):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    ox, oy = origin
    px, py = points[:, 0], points[:, 1]

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.concatenate([qx[:, None], qy[:, None]], axis=1)


def teeth_label_assignment(teeth_dict):
    clean_teeth_dict = {}
    for k, v in teeth_dict.items():
        if len(v) == 1:
            clean_teeth_dict[k] = v[0]
        else:
            for i, info in enumerate(v):
                clean_teeth_dict[f'{k}_{i}'] = info

    return clean_teeth_dict