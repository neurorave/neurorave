import torch
import torch.nn as nn
from effortless_config import Config
import logging
from termcolor import colored

import numpy as np
import math
import os

from raving_fader.pipelines import FadeRAVEPipeline
from raving_fader.realtime.config import create_config
from raving_fader.realtime.resample import Resampling
from raving_fader.realtime.trace import TraceModel

import cached_conv as cc

cc.use_cached_conv(True)


def load_model(model, ckpt):
    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"],
                                  strict=False)
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"],
                                  strict=False)


model_name = "FRAVE_JPN"

model_path = "../models/" + model_name + "/"
ckpt = model_path + model_name + "__vae_stage2.ckpt"
config_path = model_path + "train_config.yaml"
features_path = model_path + "features.pth"

config = create_config(config_path)
config.data.preprocessed = ''
pipeline = FadeRAVEPipeline(config, inst_loaders=False)

model = pipeline.model.cpu()

load_model(model, ckpt)

model.decoder.eval()
model.encoder.eval()

logging.info("flattening weights")
for m in model.modules():
    if hasattr(m, "weight_g"):
        nn.utils.remove_weight_norm(m)

logging.info("warmup forward pass")
x = torch.zeros(1, 1, 2**14)
if model.pqmf is not None:
    x = model.pqmf(x)

z, _ = model.reparametrize(*model.encoder(x))

attr = torch.zeros((z.shape[0], len(model.descriptors), z.shape[-1]))

z_c = torch.cat((z, attr), dim=1)

y = model.decoder(z_c)

if model.pqmf is not None:
    y = model.pqmf.inverse(y)

model.discriminator = None

sr = model.sr
target_sr = sr

logging.info("build resampling model")
resample = Resampling(target_sr, sr)
x = torch.zeros(1, 1, 2**14)
resample.to_target_sampling_rate(resample.from_target_sampling_rate(x))

logging.info("script model")
model = TraceModel(model, resample, features_path)
model(x, attr)

model = torch.jit.script(model)
# logging.info("../models/" + model_name + ".ts")
model.save("../models/" + model_name + ".ts")