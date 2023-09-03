from PIL import Image
import numpy as np
import torch
from stylegan2.dataset import *
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
# from test import Yolo, Segmentation, sigmoid
from pyutils import loose_bbox

def noise():
    from train_less_cgan import mixing_noise
    from cgan import TeethGenerator
    model = TeethGenerator(256, 512, 8).cuda()
    ckpt_down = torch.load('/mnt/share/shenfeihong/weight/smile-sim/2023.1.19/230000.pt', map_location=lambda storage, loc: storage)
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
    model = TeethGenerator(256, 256, 1).cuda()
    ckpt_down = torch.load('/mnt/share/shenfeihong/weight/smile-sim/2023.1.19/230000.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt_down["g_ema"])
    yolo = Yolo('/mnt/share/shenfeihong/weight/pretrain/yolo.onnx', (640, 640))
    seg = Segmentation('/mnt/share/shenfeihong/weight/pretrain/edge.onnx', (256, 256))
    sample_dir = '/mnt/share/shenfeihong/tmp/test/40photo'
    sample_dir = '/mnt/share/shenfeihong/tmp/test/40photo'
    
    save_path = '/mnt/share/shenfeihong/weight/smile-sim/2023.1.19/test'
    for i in range(100):
        sample_z = torch.randn((1,256)).cuda()
        # sample_z = torch.load(f'{save_path}/pth/95.pth').cuda()
        for file in os.listdir(sample_dir):
            img_path = os.path.join(sample_dir,file)
            # img_path = './577535.jpg'
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
            # cv2.imwrite(f"{save_path}/mask.png", mask.astype(np.uint8)*255)
            big_mask = cv2.dilate(mask, kernel=np.ones((3, 3)))
            # mask = cv2.dilate(mask, kernel=np.ones((23, 23)))-big_mask
            mask = torch.from_numpy(big_mask.astype(np.float32)[None][None]).cuda()
            mouth_tensor = mouth/255*2-1
            mouth_tensor = torch.from_numpy(mouth_tensor.transpose(2,0,1).astype(np.float32)[None]).cuda()

            sample, _ = model([sample_z], real_image=mouth_tensor, mask=1-mask, input_img=True)
            sample = sample[0].detach().cpu().numpy().transpose(1,2,0)*255/2+255/2
            # sample = big_mask[...,None]*sample+(1-big_mask[...,None])*mouth
            image[y: y + 256, x: x + 256] = sample.clip(0,255)
            img_name = img_path.split('/')[-1].split('.')[0]
            
            cv2.imwrite(f"{save_path}/{img_name}.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
            # torch.save(sample_z.detach().cpu(),f'{save_path}/pth/{i}.pth')
            # break
        
            # cv2.imwrite(f"{save_path}/{img_name}.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
        break
        
def test_single():
    from cgan import TeethGenerator
    model = TeethGenerator(256, 256, 8).cuda()
    ckpt_down = torch.load('./2022.12.13/wo_edge/040000.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt_down["g_ema"])
    yolo = Yolo('/mnt/share/shenfeihong/weight/pretrain/yolo.onnx', (640, 640))
    seg = Segmentation('/mnt/share/shenfeihong/weight/pretrain/edge.onnx', (256, 256))
    sample_dir = '/mnt/share/shenfeihong/tmp/test/40photo'
    save_path = '/mnt/share/shenfeihong/weight/smile-sim/2022.12.20/test'


    img_path = '/home/disk/data/smile_sim/smile_ffhq_2000+_seg/00086/mouth.jpg'
    image = cv2.imread(img_path)
    mouth = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    result = seg.predict(mouth)

    mask = (result[..., 0] > 0.6).astype(np.float32)
    # big_mask = cv2.dilate(mask, kernel=np.ones((3, 3)))
    mask = torch.from_numpy(mask.astype(np.float32)[None][None]).cuda()
    mouth_tensor = mouth/255*2-1
    mouth_tensor = torch.from_numpy(mouth_tensor.transpose(2,0,1).astype(np.float32)[None]).cuda()

    sample, _ = model(1, real_image=mouth_tensor, mask=1-mask, input_img=True)
    sample = sample[0].detach().cpu().numpy().transpose(1,2,0)*255/2+255/2

    img_name = img_path.split('/')[-1].split('.')[0]
    
    cv2.imwrite(f"{save_path}/1.png",cv2.cvtColor(sample, cv2.COLOR_RGB2BGR).astype(np.uint8))


def find_case():
    path = '/mnt/e/data/smile/to_b/20220713_SmileyTest_200case'
    for folder in os.listdir(path):
        img_path = os.path.join(path, folder, 'smiley.jpg')
        img = cv2.imread(img_path)
        cv2.imwrite(f'./example/ll/{folder}.jpg', img)
        
def copy_file():
    import natsort
    path = '/mnt/d/data/smile/Teeth_simulation_10K'
    for folder in natsort.natsorted(os.listdir(path))[:3000]:
        if os.path.exists(os.path.join(path, folder, 'modal', 'blend.png')):
            im = cv2.imread(os.path.join(path, folder, 'modal','mouth.png'))
            mk = cv2.imread(os.path.join(path, folder, 'modal','mouth_mask.png'))
            teeth = cv2.imread(os.path.join(path, folder, 'modal','teeth_3d.png'))
            bl = cv2.imread(os.path.join(path, folder, 'modal','blend.png'))
            
            os.makedirs(os.path.join('/mnt/e/data/smile/teeth_3d_new/8.31', folder), exist_ok=True)
            cv2.imwrite(os.path.join('/mnt/e/data/smile/teeth_3d_new/8.31', folder,'mouth.png'), im)
            cv2.imwrite(os.path.join('/mnt/e/data/smile/teeth_3d_new/8.31', folder,'mouth_mask.png'), mk)
            cv2.imwrite(os.path.join('/mnt/e/data/smile/teeth_3d_new/8.31', folder,'teeth_3d.png'), teeth)
            cv2.imwrite(os.path.join('/mnt/e/data/smile/teeth_3d_new/all', folder+'.png'), bl)
            
def select():
    path = '/mnt/e/data/smile/teeth_3d/all'
    with open('img_list.txt', 'w+')as f:
        for img_name in os.listdir(path):
            im = cv2.imread(os.path.join(path, img_name))
            cv2.imshow('im', im)
            key = cv2.waitKey()
            print(key)
            # break
            
            if key == 100:
                f.writelines(img_name.split('.')[0]+'\n')
        f.close()
        
def dilate_erode():
    import torch
    import torch.nn.functional as F

    # Create a binary image tensor (0, 1 values)
    binary_image = torch.from_numpy(cv2.imread('/mnt/d/data/smile/Teeth_simulation_10K/C01002721495/modal/mouth_mask.png').astype(np.float32))[...,-1]/255

    # Define the kernel for dilation and erosion
    dilation_kernel = torch.ones(3, 3)
    erosion_kernel = torch.ones(15, 15)

    # Perform dilation
    dilated_image = F.conv2d(binary_image.view(1, 1, *binary_image.shape), dilation_kernel.view(1, 1, *dilation_kernel.shape), padding=1)
    dilated_image = torch.where(dilated_image >= 1, torch.tensor(1.0), torch.tensor(0.0))

    # Perform erosion
    eroded_image = F.conv2d(1-binary_image.view(1, 1, *binary_image.shape), erosion_kernel.view(1, 1, *erosion_kernel.shape), padding=7)
    eroded_image = 1-torch.where(eroded_image >= 1, torch.tensor(1.0), torch.tensor(0.0))
    
    a = dilated_image[0][0].detach()-eroded_image[0][0].detach()
    cv2.imshow('dilate', 255*a.numpy().astype(np.uint8))
    cv2.waitKey(0)

def erode(binary_image):
    erosion_kernel = torch.ones(3, 3)
    eroded_image = F.conv2d(1-binary_image, erosion_kernel.view(binary_image.shape[0], 1, *erosion_kernel.shape), padding=7)
    eroded_image = 1-torch.where(eroded_image >= 1, torch.tensor(1.0), torch.tensor(0.0))
    return eroded_image
if __name__ == "__main__":
    copy_file()
    a = 1