from predict_lib import _find_objs, _seg_mouth
import tritoninferencer as tif
import cv2
import numpy as np
from io import BytesIO
import imagesize
from utils import *
from predict_lib import smile_sim_predict
def b(path) -> np.array:
    # step 0. create triton client
    triton_client = tif.create_client('0.0.0.0:8001')
    image = cv2.imread(path) 
    # step 1. find mouth obj
    objs = _find_objs(image, triton_client)
    height, width = image.shape[:2]
    mouth_objs = objs[2]
    x1, y1, x2, y2 = mouth_objs
    if x1==x2 and y1==y2:
        raise Exception("error image!")

    w, h = (x2 - x1), (y2 - y1)
    mouth_length = 256
    scale = mouth_length / max(w, h) / 1.1
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
    img_name = path.split('/')[-1]
    cv2.imwrite(f'/home/disk/data/test/smile/result/{img_name}', mouth)
    for i in range(seg_result.shape[-1]):
        cv2.imwrite(f'/home/disk/data/test/smile/result/{i}.jpg', np.array(seg_result[...,i] > 0.6, dtype=np.uint8)*255)
    return seg_result

def a(path):
    triton_client = tif.create_client('0.0.0.0:8001')
    image = cv2.imread(path) 
    objs = _find_objs(image, triton_client)
    mouth_objs = objs[2]
    print(mouth_objs)
    # x1, y1, x2, y2 = mouth_objs
    # cv2.imwrite('/home/disk/data/test/smile/result/q.jpg', image[y1:y2,x1:x2,:])
    
def dir(path):
    triton_client = tif.create_client('0.0.0.0:8001')
    with open(path, 'rb') as f:
        img_bytes = f.read()
        width, height = imagesize.get(BytesIO(img_bytes))        
        image = np.frombuffer(img_bytes, dtype = np.uint8)
        print(image[:10])

    imag = cv2.imread(path)
    print(imag[:10,...])
    # print(image.shape)
    # height, width = image
    meta = compute_meta((width, height), (640, 640), align_mode='topleft')
    
    
    res = tif.infer('cls-ensemble',
                    {'image_bin': image}, triton_client) 
    res = yolo_postprocess(res, meta,
                            iou_thr=0.2,
                            score_thr=0.3,
                            class_agnostic=False,
                            return_scores=True,
                            with_deg=True,
                            mode='xywh',
                            )

    print(res[7][0][4])
    angle = res[7][0][4]
    roundn = 0
    if angle-90>45:
        roundn = angle//90-1
        roundn += int(angle%90>45)
    print(roundn)
    img = cv2.imread(path)

    img = np.rot90(img, -roundn)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    # output2 = tif.infer('cls-yolov5s',
    #                    {'images': output['DALI_OUTPUT_0']}, triton_client)    
    # print(output2) 
    
# dir('/home/disk/data/test/smile/neg_img/20221104-144741.jpg')

import os
for file in os.listdir('/home/disk/data/test/smile/neg_img/'):
    print(file)
    smile_sim_predict(os.path.join('/home/disk/data/test/smile/neg_img/',file))
