from PIL import Image
import numpy as np
import torch
from stylegan2.dataset import *
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from test import Yolo, Segmentation, sigmoid
from tid_seg.models import get_yolo, parameter_pred, narrow_edge
from utils import loose_bbox

def noise():
    from train_less_cgan import mixing_noise
    from cgan import TeethGenerator
    from unetrain import UNet
    unet = UNet().cuda()
    unet_weight = torch.load('/mnt/share/shenfeihong/weight/smile-sim/2022.11.15/edge/555.pt')
    unet.load_state_dict(unet_weight)
    # sample_z = torch.randn((4,512)).cuda()
    
    sample_dir = '/mnt/share/shenfeihong/weight/smile-sim/2022.11.15/test'
    dataset = edge()
    loader = data.DataLoader(
        dataset,
        batch_size=6,
        sampler=data.RandomSampler(dataset),
        drop_last=True,
    )    

    iteration = iter(loader)
    for i in range(100):
        batch = next(iteration)
        mask = batch['mask'].cuda()
        edge_ = batch['edge'].cuda()
        label = batch['label'].cuda()

        input_mask = torch.cat((mask, edge_), 1)
        output = unet(input_mask)
        utils.save_image(
            output*mask,
            f"{sample_dir}/{i}.png",
            nrow=2,
            normalize=True,
            range=(0, 1),
        )
        
# torch.manual_seed(666)
# torch.cuda.manual_seed(666)
from render import *
def test_single_full():
    from edge_gan import TeethGenerator
    
    renderer = render_init()
    file_path = '/home/disk/data/tooth_arrangement/1_1000/C01001588529'
    upper, lower, mid = load_up_low(file_path, 5, show=False)
    # c_upper, c_lower, mid = load_up_low(file_path, 3, show=False)
    
    zero_ = 0
    
    model = TeethGenerator(256, 256, 8).cuda()
    ckpt_down = torch.load('/mnt/share/shenfeihong/tmp/040000.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt_down["g_ema"])
    
    # trans = cls(n_channels=3).cuda()
    # trans_weight = torch.load('/mnt/share/shenfeihong/weight/smile-sim/2022.11.23/edge/198.pt')
    # trans.load_state_dict(trans_weight)

    yolo = Yolo('/mnt/share/shenfeihong/weight/pretrain/yolo.onnx', (640, 640))
    seg = Segmentation('/mnt/share/shenfeihong/weight/pretrain/edge.onnx', (256, 256))
    tid_model = get_yolo('/mnt/share/shenfeihong/weight/pretrain')
    sample_dir = '/home/meta/sfh/data/smile/40photo'
    sample_dir = '/home/disk/data/smile_sim/tianshi/face'
    save_path = './2022.12.13/test/tianshi'
    # save_path = './2022.12.13/edge/test/'
    

    # sample_z = torch.load(f'{save_path}/_3.pth').cuda()
    for file in os.listdir(sample_dir):
        img_path = os.path.join(sample_dir,file)
        # img_path = '/home/meta/sfh/data/smile/40photo/C01008078227.jpg'
        # img_path = '/home/disk/data/smile_sim/tianshi/face/577912.jpg'
        
        print(file)
        img_name = img_path.split('/')[-1].split('.')[0]
        image = cv2.imread(img_path)
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        height, width = image.shape[:2]

        objs = yolo.predict_with_argmax(image, show=False)

        mouth_objs = objs[2]
        x1, y1, x2, y2 = mouth_objs

        w, h = (x2 - x1), (y2 - y1)

        half = max(w, h) * 1.1 / 2
        # half = max(w, h) / 2
        

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
        x1, y1, x2, y2 = loose_bbox([x1, y1, x2, y2], (width, height))
        x, y = int(x1 * 128 / half), int(y1 * 128 / half) + 2

        image = cv2.resize(image, (int(width * 128 / half), int(height * 128 / half)), cv2.INTER_AREA)
        mouth = image[y: y + 256, x: x + 256]
        result = seg.predict(mouth)

        mask = (result[..., 0] > 0.6).astype(np.float32)
        if np.sum(mask)<10:
            print('w/o mouth')
            continue
        big_mask = cv2.dilate(mask, kernel=np.ones((3, 3)))
        cir_mask = cv2.dilate(mask, kernel=np.ones((23, 23)))-big_mask

        mask = torch.from_numpy(mask.astype(np.float32)[None][None]).cuda()
        big_mask = torch.from_numpy(big_mask.astype(np.float32)[None][None]).cuda()
        cir_mask = torch.from_numpy(cir_mask.astype(np.float32)[None][None]).cuda()
        
        mouth_tensor = mouth/255*2-1
        mouth_tensor = torch.from_numpy(mouth_tensor.transpose(2,0,1).astype(np.float32)[None]).cuda()
        
        edge = (result[..., 1] > 0.6).astype(np.float32)
        up_edge = (result[..., 3] > 0.6).astype(np.float32)
        down_edge = (result[..., 2] > 0.6).astype(np.float32)
        teeth_mask = (result[..., 0] > 0.6).astype(np.float32)
        
        # os.makedirs(f"{save_path}/{img_name}", exist_ok=True)
        # cv2.imwrite(f"{save_path}/{img_name}/up_edge.png", up_edge*255)
        # cv2.imwrite(f"{save_path}/{img_name}/down_edge.png", down_edge*255)  
        # print(img_name)
        # tid = get_tid(tid_model, mouth)
        # cv2.imwrite(f"{save_path}/{img_name}/tid.png", tid)  
        # cv2.imwrite(f"{save_path}/{img_name}/mouth.png", cv2.cvtColor(mouth, cv2.COLOR_RGB2BGR))

          

        parameter = parameter_pred(mouth,up_edge,down_edge,teeth_mask,tid_model)
        if not parameter:
            continue
        dist_lower = parameter['dist']
        angle_z = parameter['anglez']
        camera_x = parameter['camerax']
        camera_y = parameter['cameray']
        camera_z = parameter['cameraz']
        x1 = parameter['x1']
        x2 = parameter['x2']
        
        
        dist_up = torch.tensor([zero_,zero_,zero_]).type(torch.float32).unsqueeze(0).cuda()
        dist_down = torch.tensor([zero_,zero_,dist_lower]).type(torch.float32).unsqueeze(0).cuda()
        angle = torch.tensor([zero_-1.35,zero_,angle_z]).type(torch.float32).unsqueeze(0).cuda()
        movement = torch.tensor([-camera_x, -camera_y, camera_z]).type(torch.float32).unsqueeze(0).cuda()

        pred_edge = render(renderer=renderer, upper=upper, lower=lower, mask=mask, angle=angle, movement=movement, dist=[dist_up, dist_down], mid=mid)

        if narrow_edge(pred_edge/255, x2-x1):
            continue
        
        pred_edge[:,:int(x1)]=0
        pred_edge[:,int(x2):]=0
        
        cv2.imwrite(f"{save_path}/edge/e_{img_name}.png", edge.astype(np.uint8)*255)
        cv2.imwrite(f"{save_path}/pred_edge/p_{img_name}.png", pred_edge)  
        
        pred_edge = torch.from_numpy((pred_edge/255).astype(np.float32)[None][None]).cuda()
        for i in range(100):
            # sample_z = (torch.randn((1,256))).cuda()   
            sample_z = torch.load(f'./2022.12.13/edge/test/pth/51.pth').cuda()
            
            # torch.save(sample_z.detach().cpu(),f'{save_path}/pth/{i}.pth')               
            sample, _ = model([sample_z], real_image=mouth_tensor, mask=cir_mask, edge=pred_edge)
            sample = big_mask*sample+(1-big_mask)*mouth_tensor
            
            sample = sample[0].detach().cpu().numpy().transpose(1,2,0)*255/2+255/2
            image[y: y + 256, x: x + 256] = sample.clip(0,255)
            cv2.imwrite(f"{save_path}/all/{img_name}.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
            break
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