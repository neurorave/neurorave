import torch
from torch import nn
import torch.nn.init as init
import numpy as np
from utils import merge_tensors

# -----------------------------------------------------------
#
# Basic CNN encoder 2D
#
# -----------------------------------------------------------


class EncoderCNN(nn.Module):
    def __init__(self, config, channels=64, n_layers=4, n_mlp=3):
        super(EncoderCNN, self).__init__()
        conv_module = nn.Conv2d
        dense_module = nn.Linear
        # Create modules
        modules = nn.Sequential()
        size = [
            config["data"]["input_size"][0], config["data"]["input_size"][1]
        ]
        hidden_size = config["model"]["enc_hidden_size"]
        out_size = config["model"]["enc_hidden_size"]
        in_channel = 1 if len(config["data"]["input_size"]) < 3 else config[
            "data"]["input_size"][0]  # in_size is (C,H,W) or (H,W)
        kernel = (11, 11)
        stride = (1, 1)
        # First do a CNN
        for layer in range(n_layers):
            dil = 1
            pad = 5  # 2
            in_s = (layer == 0) and in_channel or channels
            out_s = (layer == n_layers - 1) and 1 or channels
            modules.add_module(
                'c2%i' % layer,
                conv_module(in_s,
                            out_s,
                            kernel,
                            stride=stride,
                            padding=pad,
                            dilation=dil))
            if layer < n_layers - 1:
                modules.add_module('b2%i' % layer, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i' % layer, nn.ReLU())
                modules.add_module('d2%i' % layer, nn.Dropout2d(p=.25))
            size[0] = int((size[0] + 2 * pad -
                           (dil * (kernel[0] - 1) + 1)) / stride[0] + 1)
            size[1] = int((size[1] + 2 * pad -
                           (dil * (kernel[1] - 1) + 1)) / stride[1] + 1)
        # Then go through MLP
        self.net = modules
        self.mlp = nn.Sequential()
        for layer in range(n_mlp):
            in_s = (layer == 0) and (size[0] * size[1]) or hidden_size
            out_s = (layer == n_mlp - 1) and out_size or hidden_size
            self.mlp.add_module('h%i' % layer, dense_module(in_s, out_s))
            if layer < n_layers - 1:
                self.mlp.add_module('b%i' % layer, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i' % layer, nn.LeakyReLU())
                self.mlp.add_module('d%i' % layer, nn.Dropout(p=.25))
        self.cnn_size = size
        #self.init_parameters()

    def init_parameters(self):
        # Initialize parameters of sub-modules
        for net in [self.net, self.mlp]:
            for m in net:
                if m.__class__ in [
                        nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d,
                        nn.ConvTranspose3d
                ]:
                    init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        init.normal_(m.bias.data)
                elif m.__class__ in [
                        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
                ]:
                    init.normal_(m.weight.data, mean=1, std=0.02)
                    init.constant_(m.bias.data, 0)
                elif m.__class__ in [nn.Linear]:
                    init.xavier_normal_(m.weight.data)
                    init.normal_(m.bias.data)

    def forward(self, inputs, config):
        out = inputs.unsqueeze(1) if len(
            inputs.shape) < 4 else inputs  # force to (batch, C, H, W)
        for m in range(len(self.net)):
            out = self.net[m](out)
        out = out.view(inputs.shape[0], -1)
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        return torch.tanh(out)


# -----------------------------------------------------------
#
# Basic CNN decoder 1D
#
# -----------------------------------------------------------
class DecoderCNNFader(nn.Module):
    """
    Decoder takes enc_outputs and y (attribute) as inputs !
    """
    def __init__(self, config, channels=64, n_layers=4, n_mlp=2):
        super(DecoderCNNFader, self).__init__()
        conv_module = nn.ConvTranspose2d
        dense_module = nn.Linear
        # Create modules
        cnn_size = [
            config["model"]["cnn_size"][0], config["model"]["cnn_size"][1]
        ]
        self.cnn_size = cnn_size
        size = config["model"]["cnn_size"]
        kernel = (11, 11)
        stride = (1, 1)
        in_size = config["model"]["latent_size"] + config["data"][
            "num_attributes"]
        hidden_size = config["model"]["dec_hidden_size"]
        self.mlp = nn.Sequential()
        out_size = [
            config["data"]["input_size"][0], config["data"]["input_size"][1]
        ]
        # First go through MLP
        for layer in range(n_mlp):
            in_s = (layer == 0) and in_size or hidden_size
            out_s = (layer == n_mlp - 1) and np.prod(cnn_size) or hidden_size
            self.mlp.add_module('h%i' % layer, dense_module(in_s, out_s))
            if layer < n_layers - 1:
                self.mlp.add_module('b%i' % layer, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i' % layer, nn.ReLU())
                self.mlp.add_module('d%i' % layer, nn.Dropout(p=.25))
        modules = nn.Sequential()
        # Then do a CNN
        for layer in range(n_layers):
            dil = 1
            pad = 5
            out_pad = 0  #"(pad % 2)
            in_s = (layer == 0) and 1 or channels
            out_s = (layer == n_layers - 1) and 1 or channels
            modules.add_module(
                'c2%i' % layer,
                conv_module(in_s,
                            out_s,
                            kernel,
                            stride,
                            pad,
                            output_padding=out_pad,
                            dilation=dil))
            if layer < n_layers - 1:
                modules.add_module('b2%i' % layer, nn.BatchNorm2d(out_s))
                modules.add_module('r2%i' % layer, nn.LeakyReLU())
                modules.add_module('a2%i' % layer, nn.Dropout2d(p=.25))
            size[0] = int((size[0] - 1) * stride[0] - (2 * pad) + dil *
                          (kernel[0] - 1) + out_pad + 1)
            size[1] = int((size[1] - 1) * stride[1] - (2 * pad) + dil *
                          (kernel[1] - 1) + out_pad + 1)
        modules.add_module('final', nn.ELU())
        self.net = modules
        self.out_size = out_size  # (H,W) or (C,H,W)
        #self.init_parameters()

    def init_parameters(self):
        # Initialize internal parameters (sub-modules)
        for net in [self.net, self.mlp]:
            for m in net:
                if m.__class__ in [
                        nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d,
                        nn.ConvTranspose3d
                ]:
                    init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        init.normal_(m.bias.data)
                elif m.__class__ in [
                        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
                ]:
                    init.normal_(m.weight.data, mean=1, std=0.02)
                    init.constant_(m.bias.data, 0)
                elif m.__class__ in [nn.Linear]:
                    init.xavier_normal_(m.weight.data)
                    init.normal_(m.bias.data)

    def forward(self, inputs, y):
        out = merge_tensors(inputs, y, "cat")
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        out = out.unsqueeze(1).view(-1, 1, self.cnn_size[0], self.cnn_size[1])
        for m in range(len(self.net)):
            out = self.net[m](out)
        return out.squeeze()  # B * W * H
