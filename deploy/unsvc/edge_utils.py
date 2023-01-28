import numpy as np
import cv2
import torch

camera_dict= {
    'z_init':400,
    'z_init_width':33,
    'z_change':6.8689/90,
    'y_init_11_low':145,
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

def parameter_pred(edgeu, edged, mask, tid, mid):

    threshold = [3, None, None]

    

    if np.sum(edgeu)<threshold[0]:
        pass
    if np.sum(edged)<threshold[0]:
        pass

    # mask width height ratio to pass narrow mouth
    left_mask, right_mask = mask_width(mask)
    up_mask, down_mask = mask_length(mask)
    
    pos_up_mask = up_mask[up_mask>0]

    pos_left_mask = left_mask[left_mask>0]
    pos_right_mask = right_mask[right_mask>0]
    
    _mask_left_x = min(pos_left_mask)
    _mask_right_x = max(pos_right_mask)
    
    _mask_height = max(down_mask[up_mask>0]-up_mask[up_mask>0])
    _mask_width = _mask_right_x - _mask_left_x
    
    # if _mask_height<20 or _mask_width/_mask_height>7:
    #     print('narrow')
    #     return None

    # ori region to cut unnecessary rendere teeth 
    edgeu_left, edgeu_right = mask_width(edgeu)
    edged_left, edged_right = mask_width(edged)
    
    pos_edgeu_left = edgeu_left[edgeu_left>0]
    pos_edgeu_right = edgeu_right[edgeu_right>0]
    pos_edged_left = edged_left[edged_left>0]
    pos_edged_right = edged_right[edged_right>0]
    
    edgeu_left_min = 1e6 if len(pos_edgeu_left)==0 else np.min(pos_edgeu_left)
    edged_left_min = 1e6 if len(pos_edged_left)==0 else np.min(pos_edged_left)
    
    edgeu_right_max = 0 if len(pos_edgeu_right)==0 else np.max(pos_edgeu_right)
    edged_right_max = 0 if len(pos_edged_right)==0 else np.max(pos_edged_right)
    
    left_edge = min(edgeu_left_min, edged_left_min)
    right_edge = max(edgeu_right_max, edged_right_max)
    
    if left_edge==1e6 or right_edge==0:
        return None   

    # camera angle z
    tanh = (up_mask[int(_mask_left_x+1)] - up_mask[int(_mask_right_x-1)]) / (_mask_right_x - _mask_left_x)
    angle_z = np.arctan(tanh)
    
    # mesh movement based front teeth width
    tooth = np.zeros_like(tid)
    tooth[tid==11]=1
    if np.sum(tooth)<10:
        print('w/o 11')
        tooth = np.zeros_like(tid)
        tooth[tid==21]=1
        if np.sum(tooth) < 10:
            print('w/o 21')
            return None

    tooth_left, tooth_right = mask_width(tooth)
    _, tooth_down = mask_length(tooth)
    
    pos_tooth_left = tooth_left[tooth_left>0]
    pos_tooth_right = tooth_right[tooth_right>0]
    pos_tooth_down = tooth_down[tooth_down>0]
    
    width_11 = np.mean(pos_tooth_right[int(len(pos_tooth_right)/3):int(len(pos_tooth_right)/3*2)])\
               -np.mean(pos_tooth_left[int(len(pos_tooth_left)/3):int(len(pos_tooth_left)/3*2)])
    width_11 = width_11.clip(25,32)

    camera_z = camera_dict['z_init']-(width_11-camera_dict['z_init_width']-1)/camera_dict['z_change']
    camera_y = (np.mean(pos_tooth_down[pos_tooth_down>0])-camera_dict['y_init_11_low']-(width_11-camera_dict['z_init_width'])/2)/camera_dict['y_change']
    camera_x = (mid-128)/camera_dict['y_change']


    # distance of mask to the top of upper edge and the upper edge to the down edge
    edgeu_up, edgeu_down = mask_length(edgeu)
    edged_up, _ = mask_length(edged)
    pos_edgeu_down = edgeu_down[edgeu_down>0]
    pos_edgeu_up = edgeu_up[edgeu_up>0]
    pos_edged_up = edged_up[edged_up>0]
    
    # dist_edgeu_edged = np.mean(pos_edged_up[int(len(pos_edged_up)/3):int(len(pos_edged_up)/3*2)])\
    #            -np.mean(pos_edgeu_down[int(len(pos_edgeu_down)/3):int(len(pos_edgeu_down)/3*2)])
    dist_edgeu_edged = np.mean(pos_edged_up)\
               -np.mean(pos_edgeu_down)
    # dist_mask_edgeu = np.mean(pos_edgeu_up[int(len(pos_edgeu_up)/3):int(len(pos_edgeu_up)/3*2)])\
    #            -np.mean(pos_up_mask[int(len(pos_up_mask)/3):int(len(pos_up_mask)/3*2)])+2
    dist_mask_edgeu = (np.mean(pos_edgeu_up)\
               -np.mean(pos_up_mask))*1.1
    camera_y = camera_y - dist_mask_edgeu/camera_dict['y_change'] # move the upper edge to the top of mask 
    
    if dist_edgeu_edged>threshold[0]:
        dist_edgeu_edged = dist_edgeu_edged+camera_dict['y_init_11_low']-camera_dict['y_init_31_up']
        dist_edgeu_edged = dist_edgeu_edged/camera_dict['y_change']
    else:
        dist_edgeu_edged = 0
  
    dist_edgeu_edged = dist_edgeu_edged+dist_mask_edgeu/camera_dict['y_change']

    return {'camerax':camera_x, 'cameray':camera_y, 'cameraz':camera_z, 
            'dist':dist_edgeu_edged, 'anglez': angle_z, 'x1':left_edge, 'x2':right_edge}


def _edge_pred(parameter, mask):
    from render_utils import render
    
    dist_lower = parameter['dist']
    angle_z = parameter['anglez']
    camera_x = parameter['camerax']
    camera_y = parameter['cameray']
    camera_z = parameter['cameraz']
    x1 = parameter['x1']
    x2 = parameter['x2']
        
    zero_ = 0
    dist_up = torch.tensor([zero_,zero_,zero_]).type(torch.float32).unsqueeze(0)
    dist_down = torch.tensor([zero_,zero_,dist_lower]).type(torch.float32).unsqueeze(0)
    angle = torch.tensor([zero_-1.35,zero_,angle_z]).type(torch.float32).unsqueeze(0)
    movement = torch.tensor([-camera_x, -camera_y, camera_z]).type(torch.float32).unsqueeze(0)

    pred_edge = render.para_edge(mask=mask, angle=angle, movement=movement, dist=[dist_up, dist_down], x1=x1,x2=x2)
    pred_edge = cv2.dilate(pred_edge, np.ones((3,3))) 

    if narrow_edge(pred_edge/255, x2-x1):
        return None
    return pred_edge

def narrow_edge(pred_edge, ori_width):
    left_mask, right_mask = mask_width(pred_edge)
    
    pos_left_mask = left_mask[left_mask>0]
    pos_right_mask = right_mask[right_mask>0]
    
    _mask_left_x = np.min(pos_left_mask)
    _mask_right_x = np.max(pos_right_mask)
    ratio = (ori_width-_mask_right_x+_mask_left_x)/ori_width
    
    if ratio>0.2:
        return 1
    else:
        return 0
    

