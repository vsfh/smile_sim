import cv2
import numpy as np
from PIL import Image

rng = np.random.RandomState(666)
COLORS = rng.randint(150, 255, (37, 3))


def sigmoid(x):
    return np.exp(x) / (1+np.exp(x))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def load_image(file, color_type='rgb'):
    if color_type == 'gray':
        image = Image.open(file).convert('L').convert('RGB')
    else:
        image = Image.open(file).convert('RGB')

    image = np.array(image)
    return image


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


def preprocess_image(image, side_length, resize_ratio=1., roi=None):
    if isinstance(side_length, int):
        side_length = (side_length, side_length)
    new_image = image.copy()

    if roi is not None:
        x1, y1, x2, y2 = list(map(int, roi[:4]))
        image = new_image[y1:y2, x1:x2]
    else:
        x1, y1 = 0, 0
        image = new_image


    image, meta = resize_and_pad(image, side_length, resize_ratio=resize_ratio)

    return image, (*meta, (x1, y1))


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

def normalize_img(img):
    img = img.astype(np.float32) / 255
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def visualize_heatmap(img, heatmap, alpha=0.5):
    if len(heatmap.shape) == 3:
        heatmap = heatmap[0]
    height, width = img.shape[:2]

    heatmap[heatmap < 0] = 0.
    heatmap = (heatmap / (heatmap.max() + 1e-7) * 255).astype(np.uint8)

    seg_map = np.zeros((height, width, 3), dtype=np.uint8)
    seg_map[..., 2] = cv2.resize(heatmap, (width, height))

    img[seg_map > 50] = seg_map[seg_map > 50]
    img = img.astype(np.uint8)
    return img