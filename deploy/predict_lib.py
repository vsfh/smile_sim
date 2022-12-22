import utils
import numpy as np
import cv2
import tritoninferencer as tif
from io import BytesIO
from edge_utils import parameter_pred, _edge_pred

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

def _face_mid(image, show=False):
    from face_mid import pipeline
    import asyncio
    pipeline.show = show
    pred_coro = pipeline.predict_image_async(image)
    res = asyncio.run(pred_coro)
    # v1 = res['ly']['Ls']
    # v2 = res['ly']['Li']
    
    # target_y = (res['ly']['Ch(L)'][1]+res['ly']['Ch(R)'][1])/2
    # target_x = v1[0]+(v2[0]-v1[0])*(target_y-v1[1])/(v2[1]-v1[1])
    target_x = res['ly']['Ls'][0]
    return target_x
    
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

    # if len(res)<8:
    #     raise Exception("error image rot")
    # elif len(res[7])<1:
    #     raise Exception("error image rot")
    # elif len(res[7][0])<5:
    #     raise Exception("error image rot")
    if len(res)<8:
        return 0
    elif len(res[7])<1:
        return 0
    elif len(res[7][0])<5:
        return 0
    angle = res[7][0][4]
    roundn = 0
    angle = angle -90
    if angle < 0:
        angle += 360
        
    roundn = int(round(angle / 90)) % 4
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
    
    half = max(w, h) * 1.1 / 2
    # half = max(w, h) / 2
    

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    x1, y1, x2, y2 = utils.loose_bbox([x1, y1, x2, y2], (width, height))
    x, y = int(x1 * 128 / half), int(y1 * 128 / half) + 2

    template = cv2.resize(image, (int(width * 128 / half), int(height * 128 / half)), cv2.INTER_AREA)
    
    mid_x = _face_mid(template, False)
    mouth = template[y: y + 256, x: x + 256]
    
    if mouth.shape[0]!=256 or mouth.shape[1]!=256:
        return cv2.resize(template, (width, height))

    # step 2. edgenet seg mouth
    seg_result = _seg_mouth(mouth, triton_client)

    edge = (seg_result[..., 1] > 0.6).astype(np.float32)
    up_edge = (seg_result[..., 3] > 0.6).astype(np.float32)
    down_edge = (seg_result[..., 2] > 0.6).astype(np.float32)
    teeth_mask = (seg_result[..., 0] > 0.6).astype(np.float32)
    
    if(np.sum(teeth_mask)<10):
        return cv2.resize(template, (width, height))
    
    try:
        tid = _seg_tid(mouth, server_url)
        parameter = parameter_pred(up_edge,down_edge,teeth_mask,tid,mid_x-x)
    except:
        parameter = None
       
    big_mask = cv2.dilate(teeth_mask, kernel=np.ones((3, 3)))
    cir_mask = cv2.dilate(teeth_mask, kernel=np.ones((23, 23)))-big_mask
    cir_mask = cir_mask[...,None]
    big_mask = big_mask[...,None]

    input_image = mouth.astype(np.float32) / 255 * 2 - 1
    
    # a = np.zeros((256,256,3))
    # a[...,0] = mask[...,0]*255
    # a[...,1] = edge[...,0]*255
    # cv2.imshow('fg', a.astype(np.uint8))
    # cv2.imshow('edge', edge.astype(np.uint8)*255)
    # cv2.waitKey(0) 
    if not parameter:
        # cir_mask = cv2.dilate(teeth_mask, kernel=np.ones((33, 33)))[...,None]-big_mask
        input_dict = {'input_image':input_image,'mask':cir_mask,'big_mask':big_mask}
        network_name = 'new_smile_wo_edge_gan'
    else:
        edge = _edge_pred(parameter, teeth_mask)
        if edge is not None:
            edge = edge[...,None]/255
            input_dict = {'input_image':input_image,'mask':cir_mask,'edge':edge,'big_mask':big_mask}
            network_name = 'smile_sim_lip_preserve-up_net'
        else:
            # cir_mask = cv2.dilate(teeth_mask, kernel=np.ones((33, 33)))[...,None]-big_mask
            input_dict = {'input_image':input_image,'mask':cir_mask,'big_mask':big_mask}
            network_name = 'new_smile_wo_edge_gan'
            
    aligned_mouth = _gan(input_dict, network_name, triton_client)
    aligned_mouth = aligned_mouth.clip(0,1)*255
    aligned_mouth = aligned_mouth.astype(np.uint8)
    
    template[y: y + 256, x: x + 256] = aligned_mouth
    mid_x = _face_mid(template, True)
    
    image = cv2.resize(template, (width, height))
    return template

if __name__=="__main__":
    import os
    path = '/home/meta/sfh/data/smile/40photo'
    for file in os.listdir(path):
        # if not os.path.isfile(os.path.join('./result', file)):
        print(file)
        file = 'BC01000739458.png'
        img_path = os.path.join(path,file)
        # img_path = '/mnt/share/shenfeihong/tmp/image (8).png'
        image = cv2.imread(img_path)
        rgb_image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # _, rot_image = cv2.imencode('.jpg',image)
        server_url = '0.0.0.0:8001'
        
        with open(img_path, 'rb') as f:
            rot_image = f.read()
        output = smile_sim_predict(rot_image, rgb_image, server_url)
        output = np.array(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        
        # cv2.imwrite(os.path.join('./result', file), output)
        cv2.imshow('output', output)
        cv2.waitKey(0)
        break