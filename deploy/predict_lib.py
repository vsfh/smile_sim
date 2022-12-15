import utils
import numpy as np
import cv2
import tritoninferencer as tif
from io import BytesIO
from edge_utils import _edge_pred

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

    meta = utils.compute_meta((width, height), (640, 640), align_mode='topleft')
    
    image = np.frombuffer(image, dtype = np.uint8)
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
    from tid_models import get_tid, get_yolo
    teeth_model = get_yolo(triton_client)
    tid = get_tid(teeth_model=teeth_model, img=image)

    return tid
   
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
    roundn = face_rot(rot_image, triton_client, height, width)
    image = np.rot90(image, -roundn)
    height, width = image.shape[:2]

    # step 1. find mouth obj
    objs = _find_objs(image, triton_client)

    mouth_objs = objs[2]
    x1, y1, x2, y2 = mouth_objs
    if x1==x2 and y1==y2:
        raise Exception("error image!")

    w, h = (x2 - x1), (y2 - y1)
    w = w*1.25
    mouth_length = 256
    scale = mouth_length / max(w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    cx, cy = int(cx * scale), int(cy * scale)
    template_width, template_height = int(width * scale), int(height * scale)
    template = cv2.resize(image, (template_width, template_height))
    half = mouth_length//2
    mouth = template[cy - half:cy + half, cx - half:cx + half]
    if mouth.shape[0]!=256 or mouth.shape[1]!=256:
        return template

    # step 2. edgenet seg mouth
    seg_result = _seg_mouth(mouth, triton_client)
    fg = seg_result[..., 0]
    fg = np.array(fg > 0.6, dtype=np.float32)
    
    if(np.sum(fg)<10):
        return cv2.resize(template, (width, height))
    
    tid = _seg_tid(mouth, server_url)
    up_edge = (seg_result[..., 3] > 0.6).astype(np.float32)
    down_edge = (seg_result[..., 2] > 0.6).astype(np.float32)
    edge = _edge_pred(tid, up_edge, down_edge, cv2.erode(fg, np.ones((5,5)))).astype(np.float32)
       
    big_mask = cv2.dilate(fg, kernel=np.ones((3, 3)))
    mask = cv2.dilate(fg, kernel=np.ones((23, 23)))-big_mask
    mask = mask[...,None]
    big_mask = big_mask[...,None]
    edge = edge[...,None]/255
    input_image = mouth.astype(np.float32) / 255 * 2 - 1
    a = np.zeros((256,256,3))
    a[...,0] = mask[...,0]*255
    a[...,1] = edge[...,0]*255
    cv2.imshow('fg', a.astype(np.uint8))
    cv2.imshow('edge', edge.astype(np.uint8)*255)
    
    cv2.waitKey(0) 
    input_dict = {'input_image':input_image,'mask':mask,'edge':edge,'big_mask':fg[...,None]}
    network_name = 'smile_sim_lip_preserve-up_net'
    aligned_mouth = _gan(input_dict, network_name, triton_client)
    aligned_mouth = aligned_mouth.clip(0,1)*255
    aligned_mouth = aligned_mouth.astype(np.uint8)
    # mask = cv2.dilate(fg, kernel=np.ones((7, 7)))
    # mask = fg[...,None].astype(np.float32)
    # sample = mask*aligned_mouth+template[cy - half:cy + half, cx - half:cx + half]*(1-mask)
    template[cy - half:cy + half, cx - half:cx + half] = aligned_mouth
    image = cv2.resize(template, (width, height))
    return template

if __name__=="__main__":
    import os
    path = '/home/meta/sfh/data/smile/40photo'
    for file in os.listdir(path)[-1:]:
        if not os.path.isfile(os.path.join('./result', file)):
            print(file)
            img_path = os.path.join(path,file)
            img_path = '/home/meta/sfh/data/smile/40photo/BC01000347150.jpg'
            image = cv2.imread(img_path)
            rgb_image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # _, rot_image = cv2.imencode('.jpg',image)
            server_url = '0.0.0.0:8001'
            
            with open(img_path, 'rb') as f:
                rot_image = f.read()
            output = smile_sim_predict(rot_image, rgb_image, server_url)
            output = np.array(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            
            cv2.imshow('img', output)
            cv2.waitKey(0)
            break