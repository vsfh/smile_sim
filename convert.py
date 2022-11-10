import torch
import onnx
# from onnxsim import simplify
from encoders.psp_encoders import GradualStyleEncoder
from train_less_encoder import StyleEncoder, TeethGenerator

class alignModel(torch.nn.Module):
    def __init__(self, encoder_weight, decoder_weight, type='up'):
        super(alignModel, self).__init__()
        self.batch = False
        if type=='up':
            self.encoder = GradualStyleEncoder(50, 'ir_se')
        else:
            self.encoder = StyleEncoder(256, 256)
        # self.encoder.load_state_dict(torch.load(encoder_weight))

        self.decoder = TeethGenerator(256, 256, n_mlp=8, num_labels=2, with_bg=True)
        # ckpt_down = torch.load(decoder_weight, map_location=lambda storage, loc: storage)
        # self.decoder.load_state_dict(ckpt_down["g_ema"])

    def forward(self, x, geometry, background, mask):
        with torch.no_grad():
            codes = self.encoder(x)
            image, _ = self.decoder([codes],
                                    geometry=geometry,
                                    background=background,
                                    mask=mask,
                                    input_is_latent=True,
                                    randomize_noise=False,
                                    return_latents=True)
        return image


def convert_to_onnx(args):
    dynamic_axes = {
        'input_image': {0: 'batch_size'},
        'geometry': {0: 'batch_size'},
        'background': {0: 'batch_size'},
        'mask': {0: 'batch_size'},
        'align_img': {0: 'batch_size'}
    }


    # input = './smile/C01001459133.jpg'
    input1 = torch.randn(args.batch_size, 3, 256, 256)
    input2 = torch.randn(args.batch_size, 2, 256, 256)
    input3 = torch.randn(args.batch_size, 3, 256, 256)
    input4 = torch.randn(args.batch_size, 1, 256, 256)
    model = alignModel('/home/vsfh/training/pt/used/up_encoder.pkl', '/home/vsfh/training/pt/used/up_decoder.pt')
    input_name = ['input_image', 'geometry', 'background', 'mask']
    output_name = ['align_img']
    torch.onnx.export(model, (input1, input2, input3, input4), 'bi_net.onnx', verbose=True, input_names=input_name, output_names=output_name,
                      opset_version=12, dynamic_axes=dynamic_axes)

def onnx_simplifier():
    model = onnx.load('up_net.onnx')

    model_sim, check = simplify(model,
                                skip_shape_inference=True)

    onnx.save_model(model_sim, 'up_net_sim.onnx')


import argparse
parser = argparse.ArgumentParser(description="convert")
parser.add_argument(
    "--batch_size", type=int, default=4, help="batch sizes for each gpus"
)
args = parser.parse_args()
convert_to_onnx(args)
