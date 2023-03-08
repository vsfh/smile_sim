import utils
import numpy as np
import cv2
import tritoninferencer as tif
from io import BytesIO
from edge_utils import parameter_pred

def _find_objs(image, tinf):
    yolo_input_shape = (640, 640)
    original_h, original_w = image.shape[:2]
    resized_image, meta = utils.resize_and_pad(image, yolo_input_shape)
    offsets = meta['offsets']
    scale = meta['scale']
    input_imgs = utils.normalize_img(resized_image)
    output = tinf.infer_sync('smile_sim_lip_preserve-yolov5',
                       {'images': input_imgs}, ['output'])
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
    objs = np.array(objs, dtype=int)
    return objs

def _face_mid(image, show=False):
    from face_mid import pipeline

    pipeline.show = show
    res = pipeline.predict_image(image)
    
    # target_x = res['ly']['Ls'][0]

    # v1 = res['ly']['Sn']
    # v2 = res['ly']['V2']
    
    if res['ly']['G'][1]<0:
        return 0
    
    # target_y = (res['ly']['Ch(L)'][1]+res['ly']['Ch(R)'][1])/2
    # target_x = v1[0]+(v2[0]-v1[0])*(target_y-v1[1])/(v2[1]-v1[1])
    target_x = res['ly']['G'][0]
    return target_x

def _seg_mouth(image, tinf):
    seg_input_shape = (256, 256)
    resized_image, _ = utils.resize_and_pad(image, seg_input_shape)
    input_imgs = utils.normalize_img(resized_image)
    output = tinf.infer_sync('smile_sim_lip_preserve-edge_net',
                       {'data': input_imgs}, ['output'])
    output = np.transpose(output['output'][0], (1, 2, 0))
    output = utils.sigmoid(output)
    return output

def face_rot(image, tinf, height, width):
 
    # width, height = imagesize.get(BytesIO(image))        

    meta = utils.compute_meta((width, height), (640, 640), align_mode='topleft')
    
    image = np.frombuffer(image, dtype = np.uint8)

    res = tinf.infer_sync('cls-ensemble',
                    {'image_bin': image[None, ...]}, ['output']) 
    res = utils.yolo_postprocess(res, meta,
                            iou_thr=0.2,
                            score_thr=0.3,
                            class_agnostic=False,
                            return_scores=True,
                            with_deg=True,
                            mode='xywh',
                            )
    if len(res)<8:
        print('error image res')
        return 0
    elif len(res[7])<1:
        print('error image res')
        return 0
    elif len(res[7][0])<5:
        print('error image res')
        return 0

    angle = res[7][0][4]
    roundn = 0
    angle = angle -90
    if angle < 0:
        angle += 360
        
    roundn = int(round(angle / 90)) % 4
    return roundn

def _seg_tid(image, tinf):
    from tid_models import get_tid, get_yolo
    teeth_model = get_yolo(tinf)
    tid = get_tid(teeth_model=teeth_model, img=image)

    return tid
   
def _gan(input_dict, network_name, tinf):
    input_dict = {k: utils.normalize_img(v) for k, v in input_dict.items()}
    output = tinf.infer_sync(network_name, input_dict, ['align_img'])
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
    height, width = image.shape[:2]
    roundn = face_rot(rot_image, tinf, height, width)
    image = np.rot90(image, -roundn)
    height, width = image.shape[:2]

    error_mes = None
    # step 1. find mouth obj
    objs = _find_objs(image, tinf)

    mouth_objs = objs[2]
    x1, y1, x2, y2 = mouth_objs
    if x1==x2 and y1==y2:
        error_mes = 'not smile image'
        print(error_mes)
        image = cv2.resize(image, (int(width), int(height)), cv2.INTER_AREA)
        
        cv2.putText(image, error_mes, (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,(255,255,255), 1, cv2.LINE_AA)
        return image, roundn

    w, h = (x2 - x1), (y2 - y1)
    
    half = max(w, h) * 1.1 / 2
    # half = max(w, h) / 2
    

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    x1, y1, x2, y2 = utils.loose_bbox([x1, y1, x2, y2], (width, height))
    x, y = int(x1 * 128 / half), int(y1 * 128 / half) + 2

    template = cv2.resize(image, (int(width * 128 / half), int(height * 128 / half)), cv2.INTER_AREA)
    mid_x = _face_mid(template, False)
    print(mid_x)
    if mid_x==0:
        error_mes = 'too close'
        print(error_mes)
        
        image = cv2.resize(template, (width, height))
        cv2.putText(image, error_mes, (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,(255,255,255), 1, cv2.LINE_AA)
        return None
    mouth = template[y: y + 256, x: x + 256]
    
    if mouth.shape[0]!=256 or mouth.shape[1]!=256:
        error_mes = 'small image'
        print(error_mes)
        
        image = cv2.resize(template, (width, height))
        cv2.putText(image, error_mes, (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,(255,255,255), 1, cv2.LINE_AA)
        return None

    # step 2. edgenet seg mouth
    seg_result = _seg_mouth(mouth, tinf)

    edge = (seg_result[..., 1] > 0.6).astype(np.float32)
    up_edge = (seg_result[..., 3] > 0.6).astype(np.float32)
    down_edge = (seg_result[..., 2] > 0.6).astype(np.float32)
    teeth_mask = (seg_result[..., 0] > 0.6).astype(np.float32)
    
    if(np.sum(teeth_mask)<10):
        error_mes = 'no teeth show'
        print(error_mes)
        
        image = cv2.resize(template, (width, height))
        cv2.putText(image, error_mes, (int(width/2), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,(255,255,255), 1, cv2.LINE_AA)
        return None
    
    contours, _ = cv2.findContours(teeth_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>1:
        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        idx = np.argmax(np.array(area))
        for k in range(len(contours)):
            if k!= idx:
                teeth_mask = cv2.drawContours(teeth_mask, contours, k, 0, cv2.FILLED)
    
    
    big_mask = cv2.dilate(teeth_mask, kernel=np.ones((3, 3)))
    cir_mask = cv2.dilate(teeth_mask, kernel=np.ones((23, 23)))-big_mask
    cir_mask = cir_mask[...,None]
    big_mask = big_mask[...,None]
    



    input_image = mouth.astype(np.float32) / 255 * 2 - 1
    from edge_utils import _edge_pred, parameter_pred
    try:
        tid = _seg_tid(mouth, tinf)
        print('mouth_pos:',x,y)
        parameter = parameter_pred(up_edge,down_edge,teeth_mask,tid, mid_x-x)
    except:
        parameter = None
    if parameter is None:
        error_mes = 'teeth blur'
        input_dict = {'input_image':input_image,'mask':cir_mask,'big_mask':big_mask}
        network_name = 'new_smile_wo_edge_gan'
    else:
        try:
            edge = _edge_pred(parameter, teeth_mask)
        except:
            edge = None

        if edge is not None:
            
            edge = edge[...,None]/255
            input_dict = {'input_image':input_image,'mask':cir_mask,'edge':edge,'big_mask':big_mask}
            network_name = 'smile_sim_lip_preserve-up_net'
        else:
            error_mes = 'render fail'
            input_dict = {'input_image':input_image,'mask':big_mask}
            network_name = 'new_smile_wo_edge_gan'

    # input_dict = {'input_image':input_image,'mask':big_mask}
    # network_name = 'new_smile_wo_edge_gan'            
    aligned_mouth = _gan(input_dict, network_name, tinf)
    aligned_mouth = aligned_mouth.clip(0,1)*255
    aligned_mouth = aligned_mouth.astype(np.uint8)
    
    template[y: y + 256, x: x + 256] = aligned_mouth
    image = cv2.resize(template, (width, height))
    # mid_x = _face_mid(template, True)
    cv2.imshow('mask', np.array(big_mask))
    cv2.imwrite('a.jpg', np.array(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))
    
    # cv2.imshow('img', np.array(cv2.cvtColor(template, cv2.COLOR_RGB2BGR)))
    cv2.waitKey(0)
    return None
    
    image = cv2.resize(template, (width, height))
    if not error_mes is None:
        print(error_mes)
        cv2.putText(image, error_mes, (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,(255,255,255), 1, cv2.LINE_AA)
        
        return None
    return aligned_mouth, image

    
if __name__=="__main__":
    from tritoninferencer import TritonInferencer
    import os
    path = '/home/disk/data/smile_sim/cvat/face'
    path = '/home/meta/sfh/data/smile/40photo'
    for file in os.listdir(path):
        print(file)
        img_path = os.path.join(path,file)
        img_path = '/home/meta/sfh/data/smile/40photo/77.png'
        img_path = '/home/meta/下载/1.jpg'
        img_path = '/home/disk/data/smile_sim/align_pair/face/b.jpg'
        # img_path = '/home/disk/data/smile_sim/cvat/face/BC01000781848_photo_微笑像.JPG'
        
        image = cv2.imread(img_path)
        rgb_image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        server_url = '0.0.0.0:8001'
        tinf = TritonInferencer(server_url)
        
        with open(img_path, 'rb') as f:
            rot_image = f.read()
        output = smile_sim_predict(rot_image, rgb_image, server_url)
        if output is not None:
            
            align_image = np.array(cv2.cvtColor(output[0], cv2.COLOR_RGB2BGR))
            align_face = np.array(cv2.cvtColor(output[1], cv2.COLOR_RGB2BGR))
            
            img_name = img_path.split('/')[-1][:13]
            os.makedirs(f'/home/disk/data/smile_sim/align_pair/pair/{img_name}', exist_ok=True)
            before_image = cv2.imread(os.path.join('/home/disk/data/smile_sim/cvat/face_seg_22_11_25' , img_name, 'mouth.png'))
            before_mask = cv2.imread(os.path.join('/home/disk/data/smile_sim/cvat/face_seg_22_11_25', img_name, 'mask.png'))
            cv2.imwrite(os.path.join(f'/home/disk/data/smile_sim/align_pair/pair/{img_name}', 'align.png'), align_image)
            cv2.imwrite(os.path.join(f'/home/disk/data/smile_sim/align_pair/pair/{img_name}', 'mask.png'), before_mask)
            cv2.imwrite(os.path.join(f'/home/disk/data/smile_sim/align_pair/pair/{img_name}', 'mouth.png'), before_image)
            cv2.imwrite(os.path.join(f'/home/disk/data/smile_sim/align_pair/face/{img_name}.png'), align_face)
            
            
        # cv2.imshow('img', output)
        # cv2.waitKey(0)
        break