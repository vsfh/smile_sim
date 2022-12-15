import numpy as np
import cv2
import torch

camera_dict= {
    'z_init':400,
    'z_init_width':33,
    'z_change':7/90,
    'y_init_11_low':144,
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

def _edge_pred(tid, edgeu, edged, mask):
    from render_utils import render
    threshold = [3, None, None]
    edgeu_up, edgeu_down = mask_length(edgeu)
    edged_up, edged_down = mask_length(edged)
    mask_11 = np.zeros_like(tid)
    mask_11[tid==11]=1

    if np.sum(mask_11)<10:
        mask_21 = np.zeros_like(tid)
        mask_21[tid==21]=1
        if np.sum(mask_21) < 10:
            zero_ = 0
            dist_up = torch.tensor([zero_,zero_,zero_]).type(torch.float32).unsqueeze(0)
            dist_down = torch.tensor([zero_,zero_,zero_]).type(torch.float32).unsqueeze(0)
            angle = torch.tensor([zero_-1.396,zero_,zero_]).type(torch.float32).unsqueeze(0)
            movement = torch.tensor([zero_, zero_, 470]).type(torch.float32).unsqueeze(0)
            
            pred_edge = render.para_edge(mask=mask, angle=angle, movement=movement, dist=[dist_up, dist_down])
            pred_edge = cv2.dilate(pred_edge, np.ones((2,2))) 
        
            return pred_edge
        else:
            teeth_mask = mask_21
    else:
        teeth_mask = mask_11


    left_11, right_11 = mask_width(teeth_mask)
    up_11, down_11 = mask_length(teeth_mask)

    left_mask, right_mask = mask_width(mask)
    up_mask, down_mask = mask_length(mask)
    
    pos_up_mask = up_mask[up_mask>0]

    pos_left_mask = left_mask[left_mask>0]
    pos_right_mask = right_mask[right_mask>0]
    
    left_x = np.min(pos_left_mask)
    right_x = np.max(pos_right_mask)

    tanh = (up_mask[int(left_x+1)] - up_mask[int(right_x-1)]) / (right_x - left_x)
    angle_z = np.arctan(tanh)
    
    pos_left_11 = left_11[left_11>0]
    pos_right_11 = right_11[right_11>0]
    width_11 = np.mean(pos_right_11[int(len(pos_right_11)/3):int(len(pos_right_11)/3*2)])\
               -np.mean(pos_left_11[int(len(pos_left_11)/3):int(len(pos_left_11)/3*2)])
    # width_11 = np.max(right_11-left_11)
    width_11 = width_11.clip(0,35)

    camera_z = camera_dict['z_init']-(width_11-camera_dict['z_init_width']-3)/camera_dict['z_change']
    camera_y = (np.mean(down_11[down_11>0])-camera_dict['y_init_11_low']-(width_11-camera_dict['z_init_width'])/2)/camera_dict['y_change']

    camera_x = ((left_x+right_x)/2-127)/camera_dict['y_change']

    # camera_x = ((np.max(right_11))-127)/camera_dict['y_change']

    # camera_x = camera_x.clip(-2.5,2.5)
    pos_edgeu_down = edgeu_down[edgeu_down>0]
    pos_edgeu_up = edgeu_up[edgeu_up>0]
    pos_edged_up = edged_up[edged_up>0]
    
    dist = np.mean(pos_edged_up[int(len(pos_edged_up)/3):int(len(pos_edged_up)/3*2)])\
               -np.mean(pos_edgeu_down[int(len(pos_edgeu_down)/3):int(len(pos_edgeu_down)/3*2)])

    dist_mask_edgeu = np.mean(pos_edgeu_up[int(len(pos_edgeu_up)/3):int(len(pos_edgeu_up)/3*2)])\
               -np.mean(pos_up_mask[int(len(pos_up_mask)/3):int(len(pos_up_mask)/3*2)])
    print(dist_mask_edgeu)
    camera_y = camera_y-dist_mask_edgeu/camera_dict['y_change']
               
    if dist>threshold[0]:
        dist = dist+camera_dict['y_init_11_low']-camera_dict['y_init_31_up']
        dist_lower = dist/camera_dict['y_change']
    else:
        dist_lower = 0
        
    zero_ = 0
    dist_up = torch.tensor([zero_,zero_,zero_]).type(torch.float32).unsqueeze(0)
    dist_down = torch.tensor([zero_,zero_,dist_lower]).type(torch.float32).unsqueeze(0)
    angle = torch.tensor([zero_-1.3,zero_,angle_z]).type(torch.float32).unsqueeze(0)
    movement = torch.tensor([-camera_x, -camera_y, camera_z]).type(torch.float32).unsqueeze(0)

    pred_edge = render.para_edge(mask=mask, angle=angle, movement=movement, dist=[dist_up, dist_down])
    pred_edge = cv2.dilate(pred_edge, np.ones((3,3))) 

    return pred_edge


    

