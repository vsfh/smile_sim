import cv2
import numpy as np

def sigmoid(x):
    return np.exp(x) / (1+np.exp(x))

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

    if img.max() > 1.:
        img = img / 255

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img

def compute_meta(src_shape, dst_shape, resize_ratio=1., bbox=None, align_mode='center'):
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
        'resize_shape': (resize_width, resize_height)
    }
    return meta

def yolo_postprocess(outputs, meta, score_thr=0.4, iou_thr=0.4, class_agnostic=False, return_scores=False,
                        with_deg=False,
                        mode='xyxy',
                        ):
    grid = outputs['output'][0]
    yolo_xywh = grid[:, :4]

    if with_deg:
        probs = grid[:, 4:5] * grid[:, 5:-2]
        cosr, sinr = grid[:, -2], grid[:, -1]
        theta = np.arccos(cosr)
        theta = np.rad2deg(theta)
        theta[sinr < 0] = 360- theta[sinr < 0]

    else:
        probs = grid[:, 4:5] * grid[:, 5:]

    labels = np.argmax(probs, axis=1)
    scores = np.max(probs, axis=1)
    labels_set = set(labels)
    num_objs, num_classes = probs.shape[:2]

    offsets = meta['offsets']
    scale = meta['scale']
    xyxy = np.zeros_like(yolo_xywh)
    xyxy[:, 0] = yolo_xywh[:, 0] - yolo_xywh[:, 2] / 2
    xyxy[:, 1] = yolo_xywh[:, 1] - yolo_xywh[:, 3] / 2
    xyxy[:, 2] = yolo_xywh[:, 0] + yolo_xywh[:, 2] / 2
    xyxy[:, 3] = yolo_xywh[:, 1] + yolo_xywh[:, 3] / 2

    # map back
    xyxy[:, [0, 2]] -= offsets[0]
    xyxy[:, [1, 3]] -= offsets[1]
    xyxy /= scale
    xyxy[:, [0, 2]] += meta['rect'][0]
    xyxy[:, [1, 3]] += meta['rect'][1]

    # remove invalid bbox first
    width, height = meta['image_size']
    min_length = 4
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    invalid_index = (x1 >= (width - 1 - min_length)) | (y1 >= (height - 1 - min_length)) \
                    | (x2 <= min_length) | (y2 <= min_length) | \
                    ((x1 + min_length) >= x2) | ((y1 + min_length) >= y2)
    scores[invalid_index] = 0.

    xywh = xyxy.copy()
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

    if class_agnostic:
        keep = cv2.dnn.NMSBoxes(xywh, scores, score_thr, iou_thr)
    else:
        keep = []
        number = np.arange(num_objs)
        for i in range(num_classes):
            indices = labels == i
            bboxes = xywh[indices]
            cur_number = number[indices]
            cur_scores = scores[indices]
            cur_keep = cv2.dnn.NMSBoxes(bboxes, cur_scores, score_thr, iou_thr)
            if len(cur_keep) != 0:
                keep.extend(cur_number[cur_keep])

    objs = [[] for _ in range(num_classes)]
    for k in keep:
        l = labels[k]
        s = scores[k:k + 1]

        if mode == 'xyxy':
            bbox = xyxy[k]
            bbox = loose_bbox(bbox, meta['image_size'])
        else:
            x1, y1, x2, y2 = xyxy[k]
            cx, cy = (x1 + x2) /2, (y1+y2)/2
            w, h =x2-x1, y2- y1
            bbox = np.array([cx, cy, w, h])

        if with_deg:
            angle = theta[k:k+1]
            bbox = np.concatenate([bbox, angle], axis=0)

        if return_scores:
            bbox = np.concatenate([bbox, s], axis=0)

        objs[l].append(bbox)

    return objs

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