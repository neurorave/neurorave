import torch
import torch.nn as nn
from effortless_config import Config
import logging
from termcolor import colored
import numpy as np
import math

from raving_fader.models.fader.faderave import FadeRAVE
from raving_fader.realtime.resample import Resampling


class TraceModel(nn.Module):
    def __init__(self,
                 pretrained: FadeRAVE,
                 resample: Resampling,
                 features_path: str,
                 stereo: bool = False,
                 deterministic: bool = False):
        super().__init__()

        latent_size = pretrained.latent_size
        self.resample = resample

        self.pqmf = pretrained.pqmf
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        self.sr = pretrained.sr
        self.descriptors = pretrained.descriptors

        self.compute_minmax(features_path)

        self.register_buffer("latent_size", torch.tensor(latent_size))
        self.register_buffer(
            "sampling_rate",
            torch.tensor(self.resample.taget_sr),
        )
        try:
            self.register_buffer("max_batch_size",
                                 torch.tensor(cc.MAX_BATCH_SIZE))
        except:
            print(
                "You should upgrade cached_conv if you want to use RAVE in batch mode !"
            )
            self.register_buffer("max_batch_size", torch.tensor(1))
        self.trained_cropped = bool(pretrained.cropped_latent_size)
        self.deterministic = deterministic

        self.cropped_latent_size = pretrained.cropped_latent_size

        x = torch.zeros(1, 1, 2**14)
        z = self.encode(x)
        ratio = x.shape[-1] // z.shape[-1]

        self.register_buffer(
            "encode_params",
            torch.tensor([
                1,
                1,
                self.cropped_latent_size,
                ratio,
            ]))

        self.register_buffer(
            "decode_params",
            torch.tensor([
                self.cropped_latent_size,
                ratio,
                2 if stereo else 1,
                1,
            ]))

        self.register_buffer("forward_params",
                             torch.tensor([1, 1, 2 if stereo else 1, 1]))

        self.stereo = stereo

    def compute_minmax(self, features_path):
        features = torch.load(features_path)

        self.min_max_features = []
        for i, descr in enumerate(self.descriptors):
            self.min_max_features.append(
                [np.min(features[:, i, :]),
                 np.max(features[:, i, :])])

        self.min_max_features = torch.tensor(self.min_max_features)
        print(self.min_max_features)

    def post_process_distribution(self, mean, scale):
        std = nn.functional.softplus(scale) + 1e-4
        return mean, std

    def reparametrize(self, mean, std):
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return z, kl

    def normalize(self, array, min_max):
        return (2 * ((array - min_max[0]) / (min_max[1] - min_max[0]) - 0.5))

    @torch.jit.export
    def normalize_all(self, attr):
        for i, descr in enumerate(self.descriptors):
            attr[:, i] = self.normalize(attr[:, i], self.min_max_features[i])

        return (attr)

    @torch.jit.export
    def encode(self, x):
        x = self.resample.from_target_sampling_rate(x)

        if self.pqmf is not None:
            x = self.pqmf(x)

        mean, scale = self.encoder(x)
        mean, std = self.post_process_distribution(mean, scale)

        if self.deterministic:
            z = mean
        else:
            z = self.reparametrize(mean, std)[0]
        return z

    @torch.jit.export
    def encode_amortized(self, x):
        x = self.resample.from_target_sampling_rate(x)

        if self.pqmf is not None:
            x = self.pqmf(x)

        mean, scale = self.encoder(x)
        mean, std = self.post_process_distribution(mean, scale)
        var = std * std
        std = var.sqrt()

        return mean, std

    @torch.jit.export
    def decode(self, z):

        if self.stereo and z.shape[0] == 1:  # DUPLICATE LATENT PATH
            z = z.expand(2, z.shape[1], z.shape[2])

        x = self.decoder(z, add_noise=not self.deterministic)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)

        x = self.resample.to_target_sampling_rate(x)

        if self.stereo:
            x = x.permute(1, 0, 2)
        return x

    def forward(self, x, attr):
        z = self.encode(x)
        z_c = torch.cat((z, attr), dim=1)
        return self.decode(z_c)