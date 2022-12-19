import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
from encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE

class network(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(network, self).__init__()
        # print('Using BackboneEncoderRefineStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(BatchNorm2d(256),
                                         torch.nn.AdaptiveAvgPool2d((3, 3)),
                                         Flatten(),
                                         Linear(256 * 3 * 3, 256 * 7))
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((3, 3)),
                                         Flatten(),
                                         Linear(128 * 3 * 3, 256 * 4))
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((3, 3)),
                                         Flatten(),
                                         Linear(64 * 3 * 3, 256 * 3))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

    def forward(self, x):
        # x = torch.cat((x,first_stage_output_image), dim=1)
        x = self.input_layer(x)
        for l in self.modulelist[:1]:
          x = l(x)
        lc_part_4 = self.output_layer_5(x).view(-1, 3, 256)
        for l in self.modulelist[1:2]:
          x = l(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 4, 256)
        for l in self.modulelist[2:3]:
          x = l(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 7, 256)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x
      