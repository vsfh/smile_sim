from PIL import Image
import numpy as np
import torch
from stylegan2.dataset import *
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from test import Yolo, Segmentation, sigmoid
from utils import loose_bbox

def noise():
    from train_less_cgan import mixing_noise
    from cgan import TeethGenerator
    model = TeethGenerator(256, 512, 8).cuda()
    ckpt_down = torch.load('/mnt/share/shenfeihong/weight/smile-sim/2022.11.11/150000.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt_down["g_ema"])
    # sample_z = torch.randn((4,512)).cuda()
    
    sample_dir = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.8/test'
    dataset = fuck(mode='encoder')
    loader = data.DataLoader(
        dataset,
        batch_size=6,
        sampler=data.RandomSampler(dataset),
        drop_last=True,
    )    

    iteration = iter(loader)
    for i in range(100):
        batch = next(iteration)
        real_img = batch['images'].cuda()
        mask = batch['mask'].cuda()
        # code = [model.style(s) for s in [sample_z]]
        # print(code[0].shape)
        sample_z = torch.randn((6,512)).cuda()*i/100
        sample, _ = model([sample_z], real_image=real_img, mask=mask)
        utils.save_image(
            sample,
            f"{sample_dir}/{i}.png",
            nrow=2,
            normalize=True,
            range=(-1, 1),
        )
        
# torch.manual_seed(666)
# torch.cuda.manual_seed(666)
def test_single_full():
    from cgan import TeethGenerator
    model = TeethGenerator(256, 256, 8).cuda()
    ckpt_down = torch.load('/mnt/share/shenfeihong/weight/smile-sim/2022.12.2/050000.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt_down["g_ema"])
    yolo = Yolo('/mnt/share/shenfeihong/weight/pretrain/yolo.onnx', (640, 640))
    seg = Segmentation('/mnt/share/shenfeihong/weight/pretrain/edge.onnx', (256, 256))
    sample_dir = '/mnt/share/shenfeihong/data/test/11.8.2022'
    save_path = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.11/test'
    for i in range(100):
        sample_z = torch.randn((1,256)).cuda()
        # sample_z = torch.load(f'{save_path}/_3.pth').cuda()
        for file in os.listdir(sample_dir):
            img_path = os.path.join(sample_dir,file)
            image = cv2.imread(img_path)
            image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            height, width = image.shape[:2]

            objs = yolo.predict_with_argmax(image, show=False)

            mouth_objs = objs[2]
            x1, y1, x2, y2 = mouth_objs

            w, h = (x2 - x1), (y2 - y1)

            half = max(w, h) * 1.25 / 2
            # half = max(w, h) / 2
            

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
            x1, y1, x2, y2 = loose_bbox([x1, y1, x2, y2], (width, height))
            x, y = int(x1 * 128 / half), int(y1 * 128 / half) + 2

            image = cv2.resize(image, (int(width * 128 / half), int(height * 128 / half)), cv2.INTER_AREA)
            mouth = image[y: y + 256, x: x + 256]
            result = seg.predict(mouth)

            mask = (result[..., 0] > 0.6).astype(np.float32)
            big_mask = cv2.dilate(mask, kernel=np.ones((3, 3)))
            mask = cv2.dilate(mask, kernel=np.ones((33, 33)))-big_mask
            mask = torch.from_numpy(mask.astype(np.float32)[None][None]).cuda()
            mouth_tensor = mouth/255*2-1
            mouth_tensor = torch.from_numpy(mouth_tensor.transpose(2,0,1).astype(np.float32)[None]).cuda()

            sample, _ = model([sample_z], real_image=mouth_tensor, mask=mask)
            sample = sample[0].detach().cpu().numpy().transpose(1,2,0)*255/2+255/2
            sample = big_mask[...,None]*sample+(1-big_mask[...,None])*mouth
            image[y: y + 256, x: x + 256] = sample.clip(0,255)
            img_name = img_path.split('/')[-1].split('.')[0]
            
            cv2.imwrite(f"{save_path}/_{i}.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
            torch.save(sample_z.detach().cpu(),f'{save_path}/_{i}.pth')
            break
        
            # cv2.imwrite(f"{save_path}/{img_name}.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
        # break
        

        # utils.save_image(
        #     sample,
        #     f"{save_path}/{img_name}.png",
        #     nrow=2,
        #     normalize=True,
        #     range=(-1, 1),
        # )
    
if __name__ == "__main__":
    test_single_full()
    a = 1