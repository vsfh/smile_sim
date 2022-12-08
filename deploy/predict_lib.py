import utils
import numpy as np
import cv2
import tritoninferencer as tif
from io import BytesIO
from render import _edge_pred

def _find_objs(image, triton_client):
    yolo_input_shape = (640, 640)
    original_h, original_w = image.shape[:2]
    resized_image, meta = utils.resize_and_pad(image, yolo_input_shape)
    offsets = meta['offsets']
    scale = meta['scale']
    input_imgs = utils.normalize_img(resized_image)
    output = tif.infer('smile_sim_lip_preserve-yolov5',
                       {'images': input_imgs}, triton_client)
    output = output['output'][0]
    xywh = output[:, :4]
    probs = output[:, 4:5] * output[:, 5:]

    objs = []
    num_class = probs.shape[-1]

    for i in range(num_class):
        p = probs[:, i]
        if p.max() < 0.04:
            a = np.array([0,0,0,0])
            objs.append(a)
            continue
        idx = p.argmax()

        x, y, w, h = xywh[idx]
        coords = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

        coords[[0, 2]] -= offsets[0]
        coords[[1, 3]] -= offsets[1]
        coords /= scale

        coords = utils.loose_bbox(coords, (original_w, original_h))
        objs.append(coords)
    objs = np.array(objs, dtype=np.int)
    return objs


def _seg_mouth(image, triton_client):
    seg_input_shape = (256, 256)
    resized_image, _ = utils.resize_and_pad(image, seg_input_shape)
    input_imgs = utils.normalize_img(resized_image)
    output = tif.infer('smile_sim_lip_preserve-edge_net',
                       {'data': input_imgs}, triton_client)
    output = np.transpose(output['output'][0], (1, 2, 0))
    output = utils.sigmoid(output)
    return output

def face_rot(image, triton_client, height, width):
 
    # width, height = imagesize.get(BytesIO(image))        
    # image = np.frombuffer(image, dtype = np.uint8)

    meta = utils.compute_meta((width, height), (640, 640), align_mode='topleft')
    
    
    res = tif.infer('cls-ensemble',
                    {'image_bin': image}, triton_client) 
    res = utils.yolo_postprocess(res, meta,
                            iou_thr=0.2,
                            score_thr=0.3,
                            class_agnostic=False,
                            return_scores=True,
                            with_deg=True,
                            mode='xywh',
                            )
    angle = res[7][0][4]
    roundn = 0
    if angle-90>45:
        roundn = angle//90-1
        roundn += int(angle%90>45)
    return roundn

def _seg_tid(image, triton_client):
    result = tif.infer('smile_sim_lip_preserve-tid_seg',
                       {'images': image}, triton_client)
    img_show = np.zeros((256,256))
    for res in result[0]:
        if len(res['points'])==0:
            continue
        points = res['points'][0]
        bin1 = res['bin1']
        bin2 = res['bin2']

        a = np.argmax(bin1) + 1
        b = np.argmax(bin2) + 1
        fdi = int(a*10+b)
        if fdi==11 or fdi==21:
            cv2.fillPoly(img_show, pts=[points.astype(int)[:, None]], color=(fdi))
    return img_show
   
def _gan(input_dict, network_name, triton_client):
    input_dict = {k: utils.normalize_img(v) for k, v in input_dict.items()}
    output = tif.infer(network_name, input_dict, triton_client)
    output = output['align_img'][0].transpose(1, 2, 0)
    output = (output + 1.0) / 2.0
    return output

def smile_sim_predict(
    rot_image,
    image: np.array,
    server_url: str,

) -> np.array:
    '''predict function to simulate smile.

    Inputs:
        image: numpy representation of the image.
        server_url: url to server grpc port
    Outputs:
        image: numpy representation of inferred image.
    '''

    # step 0. create triton client
    triton_client = tif.create_client(server_url)
    height, width = image.shape[:2]
    # roundn = face_rot(image, triton_client, height, width)
    # image = np.rot90(image, -roundn)
    
    # step 1. find mouth obj
    objs = _find_objs(image, triton_client)

    mouth_objs = objs[2]
    x1, y1, x2, y2 = mouth_objs
    if x1==x2 and y1==y2:
        raise Exception("error image!")

    w, h = (x2 - x1), (y2 - y1)
    mouth_length = 256
    scale = mouth_length / max(w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    cx, cy = int(cx * scale), int(cy * scale)
    template_width, template_height = int(width * scale), int(height * scale)
    template = cv2.resize(image, (template_width, template_height))
    half = mouth_length // 2
    mouth = template[cy - half:cy + half, cx - half:cx + half]
    if mouth.shape[0]!=256 or mouth.shape[1]!=256:
        return template

    # step 2. edgenet seg mouth
    seg_result = _seg_mouth(mouth, triton_client)
    fg = seg_result[..., 0]
    fg = np.array(fg > 0.6, dtype=np.float32)
    
    tid = _seg_tid(mouth, triton_client)
    up_edge = (seg_result[..., 3] > 0.6).astype(np.float32)
    down_edge = (seg_result[..., 2] > 0.6).astype(np.float32)
    edge = _edge_pred(tid, up_edge, down_edge, mask)
        
    big_mask = cv2.dilate(fg, kernel=np.ones((3, 3)))
    mask = cv2.dilate(fg, kernel=np.ones((33, 33)))-big_mask
    mask = mask[...,None]
    big_mask = big_mask[...,None]
    input_image = mouth.astype(np.float32) / 255 * 2 - 1

    input_dict = {'input_image':input_image,'mask':mask,'edge':edge,'big_mask':big_mask}
    network_name = 'smile_sim_lip_preserve-up_net'
    aligned_mouth = _gan(input_dict, network_name, triton_client)
    aligned_mouth = aligned_mouth.clip(0,1)*255
    aligned_mouth = aligned_mouth.astype(np.uint8)
    mask = cv2.dilate(fg, kernel=np.ones((7, 7)))
    mask = mask[...,None].astype(np.float32)
    sample = mask*aligned_mouth+template[cy - half:cy + half, cx - half:cx + half]*(1-mask)
    template[cy - half:cy + half, cx - half:cx + half] = sample
    image = cv2.resize(template, (width, height))
    return image