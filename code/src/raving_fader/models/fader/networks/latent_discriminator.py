import torch
import torch.nn as nn
import cached_conv as cc


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_size=128, num_attributes=1, num_classes=16, num_layers=2):
        super(LatentDiscriminator, self).__init__()
        net = []
        for _ in range(num_layers):
            net.append(nn.Conv1d(
                latent_size,
                latent_size,
                7,
                stride=1,
                padding=cc.get_padding(7, mode="centered")[-1],
                bias=False,
            ))
            net.append(nn.BatchNorm1d(latent_size))
            net.append(nn.LeakyReLU(.2))

        net.append(nn.Conv1d(
                latent_size,
                latent_size // 2,
                7,
                stride=1,
                padding=cc.get_padding(7, mode="centered")[-1],
                bias=False,
            ))
        net.append(nn.BatchNorm1d(latent_size // 2))
        net.append(nn.LeakyReLU(.2))
        self.net = nn.Sequential(*net)

        attr_nets = []
        for _ in range(num_attributes):
            attr_net = []
            attr_net.append(nn.Conv1d(
                latent_size // 2,
                latent_size // 4,
                7,
                stride=1,
                padding=cc.get_padding(7, mode="centered")[-1],
                bias=False,
            ))
            attr_net.append(nn.BatchNorm1d(latent_size // 4))
            attr_net.append(nn.LeakyReLU(.2))
            attr_net.append(nn.Conv1d(
                latent_size // 4,
                num_classes,
                7,
                stride=1,
                padding=cc.get_padding(7, mode="centered")[-1],
                bias=False,
            ))
            
            attr_nets.append(nn.Sequential(*attr_net))
        self.attr_nets = nn.ModuleList(attr_nets)

    def forward(self, z):
        x = self.net(z)
        out_attr_cls = []
        for layer in self.attr_nets:
            out_attr_cls.append(layer(x))
        out_attr_cls = [t.unsqueeze(dim=-2) for t in out_attr_cls]
        out = torch.cat(out_attr_cls, dim=-2)
        return out
