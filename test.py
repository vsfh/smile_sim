import torch
# from train_less_encoder import PSP
# from encoders.psp_encoders import GradualStyleEncoder
# from cgan import TeethGenerator
import onnxruntime
from pyutils import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BaseModel:
    def __init__(self, weight_path, image_size):
        sess_option = onnxruntime.SessionOptions()
        sess_option.log_severity_level = 4

        self.weight_path = weight_path
        self.sess = onnxruntime.InferenceSession(weight_path,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.image_size = image_size
        self.input_name = self.sess.get_inputs()[0].name
        self.input_name = [n.name for n in self.sess.get_inputs()]

    # def predict(self, image, show=True):
    #     raise NotImplementedError


class Yolo(BaseModel):
    def predict_with_argmax(self, image, show=True):
        if isinstance(image, str):
            image = load_image(image)

        height, width = image.shape[:2]

        resized_image, meta = resize_and_pad(image, self.image_size)
        offsets = meta['offsets']
        scale = meta['scale']

        input_imgs = normalize_img(resized_image)
        output = self.sess.run([], {self.input_name[0]: input_imgs})
        output = output[0][0]
        xywh = output[:, :4]
        probs = output[:, 4:5] * output[:, 5:]

        objs = []
        num_class = probs.shape[-1]

        for i in range(num_class):
            p = probs[:, i]
            idx = p.argmax()

            x, y, w, h = xywh[idx]
            coords = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

            coords[[0, 2]] -= offsets[0]
            coords[[1, 3]] -= offsets[1]
            coords /= scale

            coords = loose_bbox(coords, (width, height))
            objs.append(coords)

            if show:
                x1, y1, x2, y2 = coords
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


        objs = np.array(objs, dtype=np.int64)
        return objs


class Segmentation(BaseModel):
    def predict(self, image, show=False):
        if isinstance(image, str):
            image = load_image(image)

        resized_image, meta = resize_and_pad(image, self.image_size)
        # input_imgs = resized_image.astype(np.float32)[None] / 255
        input_imgs = normalize_img(resized_image)
        output = self.sess.run([], {self.input_name[0]: input_imgs})
        output = np.transpose(output[0][0], (1, 2, 0))
        output = sigmoid(output)
        if show:
            for i in range(num_channels):
                print(output[..., i].max())


        return output

class alignModel(torch.nn.Module):
    def __init__(self, encoder_weight, decoder_weight, type='up'):
        super(alignModel, self).__init__()
        self.batch = False
        if type == 'up':
            self.encoder = GradualStyleEncoder(50, 'ir_se').cuda()
        else:
            self.encoder = PSP.load_from_checkpoint(encoder_weight).psp_encoder.cuda()
        # self.encoder.load_state_dict(torch.load(encoder_weight))

        self.decoder = TeethGenerator(256, 256, n_mlp=8, num_labels=2, with_bg=True).cuda()
        ckpt_down = torch.load(decoder_weight, map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(ckpt_down["g_ema"])

    def forward(self, x, geometry, background, mask):
        with torch.no_grad():
            # codes1 = self.encoder(x)
            codes = torch.mean(torch.randn((10000, 256)), dim=0).view(1, 256).cuda()
            codes = [codes, codes]
            image, _ = self.decoder(codes,
                                    geometry=geometry,
                                    background=background,
                                    mask=mask,
                                    input_is_latent=False,
                                    randomize_noise=False,
                                    return_latents=True)
        return image


if __name__ == '__main__':
    import os

    up_net = alignModel('./weights/up_teeth_encoder/version_6/01000.pkl',
                        '/home/vsfh/training/less_cgan/down/020000.pt', 'up')
    # bi_net = alignModel('/home/vsfh/training/less_encoder/down/less_condition_up_encoder_2021-11-03_13_38_42'
    #                     '/version_0/checkpoints/epoch=0-step=1624.ckpt', '/home/vsfh/training/less_cgan/down/020000.pt', 'bi')
    yolo = Yolo('pretrained_models/smile_yolov5xs.onnx', (640, 640))
    seg = Segmentation('pretrained_models/EdgeNet.onnx', (256, 256))

    base_path = '/home/vsfh/dataset/test_case/single'
    out_path = '/home/vsfh/dataset/test_case/out'
    flist = os.listdir(base_path)
    for file in flist:
        # if not file.endswith('jpg'):
        #     continue
        img_path = os.path.join(base_path, file)

        image = cv2.imread(img_path)
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        height, width = image.shape[:2]

        objs = yolo.predict_with_argmax(image, show=False)

        mouth_objs = objs[2]
        x1, y1, x2, y2 = mouth_objs
        loose_factor = 1.1
        w, h = (x2 - x1), (y2 - y1)

        half = max(w, h) * 1.1 / 2

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
        x1, y1, x2, y2 = loose_bbox([x1, y1, x2, y2], (width, height))
        x, y = int(x1 * 128 / half), int(y1 * 128 / half) + 2

        image = cv2.resize(image, (int(width * 128 / half), int(height * 128 / half)), cv2.INTER_AREA)
        mouth = image[y: y + 256, x: x + 256]
        result = seg.predict(mouth)
        up_teeth = False
        mouth_mask = (sigmoid(result[..., 1]) > 0.6).astype(np.uint8)
        pic = (sigmoid(result[..., 4]) > 0.6).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mouth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = 0
        for i in range(len(contours)):
            res += cv2.contourArea(contours[i])
        if res < 500:
            up_teeth = True
        mouth = mouth.astype(np.float32) / 255.0 * 2 - 1
        kernel_size = 9
        print(up_teeth)
        # print(self.src_mask_dir, mask_name)
        fg = pic.astype(np.float32)

        # fg[:40, :] = 1
        # fg[216:, :] = 1

        mask = cv2.dilate(fg, kernel=np.ones((kernel_size, kernel_size)))

        transition = mask - fg

        mask = mask.astype(np.float32)
        input_semantic = np.zeros((256, 256, 2), dtype=np.float32)
        input_semantic[..., 0] = fg
        input_semantic[..., 1] = transition

        input_semantic = input_semantic.transpose(2, 0, 1)
        bg = (1 - mask)[..., None] * mouth
        np.savez('test.npz', mouth=mouth, bg=bg, mask=mask, input_semantic=input_semantic)
        mouth = torch.FloatTensor(mouth.transpose(2, 0, 1)).unsqueeze(0).cuda()
        bg = torch.FloatTensor(bg.transpose(2, 0, 1)).unsqueeze(0).cuda()
        mask = torch.FloatTensor(mask.astype(np.float32)[None]).unsqueeze(0).cuda()
        input_semantic = torch.FloatTensor(input_semantic).unsqueeze(0).cuda()
        # if up_teeth:
        #     out = up_net(mouth, input_semantic, bg, mask)
        # else:
        out = up_net(mouth, input_semantic, bg, mask)
        sample = out.cpu().numpy()[0]
        sample = (sample + 1) / 2
        sample = sample.transpose(1, 2, 0)[..., ::-1]
        sample = np.clip(sample, 0., 1.)
        sample = (sample * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # if up_teeth:
        image[y + 45:y + 211, x:x + 256] = sample[45:211, :]
        # else:
        # for i in range(256):
        #     for j in range(176):
        #         image[y+40+j, x+i] = sample[j+40, i]
        image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_path, file), image)
