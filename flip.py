import os
import cv2


base_dir = '/home/vsfh/dataset/flip/down_mask'
out_path = '/home/vsfh/dataset/mask/a_down'
img_list = os.listdir(base_dir)

for img in img_list:
    in_dir = os.path.join(base_dir, img)
    a = cv2.imread(in_dir)
    out_dir = os.path.join(out_path, img)
    cv2.imwrite(out_dir, a[:,::-1,:])

