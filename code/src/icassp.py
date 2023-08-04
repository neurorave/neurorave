import os
import torch
import torch.nn as nn

from raving_fader.helpers.core import load_config
from raving_fader.config import settings, RaveConfig, DataConfig, TrainConfig
from raving_fader.datasets.nsynth_dataset import NSynthDataset
from raving_fader.datasets.data_loaders import nsynth_data_loader, rave_data_loaders

"""
Configuration and basic definitions if needed
"""
data_dir = "/home/hime/Work/dataset/impacts"


# Load main config file TODO Make new .yaml if needed
config = load_config("/home/hime/Work/Coding/raving-fader/config/rave_config.yaml")
# Load data confid TODO make new if needed
data_config = DataConfig(**config["data"])

# create results directory
os.makedirs(os.path.join(models_dir, name), exist_ok=True)

# add / update values
data_config.wav = os.path.join(data_dir, "audio")
data_config.preprocessed = data_dir
if sr:
    data_config.sr = int(sr)

# >>> MODEL
rave_config = RaveConfig(**config["rave"])
# rave
if warmup:
    rave_config.warmup = warmup
fader_config = None
if model == "faderave":
    fader_config = FaderConfig(rave=rave_config, **config["fader"])

# >>> TRAIN
if name:
    config["train"]["name"] = name
train_config = TrainConfig(**config["train"])
name = train_config.name
# train
train_config.models_dir = os.path.join(models_dir, name)
if batch:
    train_config.batch = batch
if max_steps:
    train_config.max_steps = max_steps
if ckpt:
    train_config.ckpt = ckpt
if lambda_inf:
    train_config.lambda_inf = lambda_inf
if rave_mode:
    train_config.rave_mode = rave_mode

# save config
config = BaseConfig(data=data_config, rave=rave_config, fader=fader_config, train=train_config)

data_config.wav = data_dir
data_config.preprocessed = os.path.join(data_dir, name, "tmp")

config = RaveConfig(data=data_config, model=model_config)

config.data.sr = 16000
n_band = 16
ratio = 4 * 4 * 4 * 2
latent_length = int(config.data.n_signal / n_band / ratio)
print(config.data.descriptors)

"""
Import data:
- make one huge repertory with all samples
- compute all attributes and save them
- return train, val
"""
dataset = get_dataset_attr(
    preprocessed=config.data.preprocessed,
    wav=config.data.wav,
    sr=config.data.sr,
    descriptors=config.data.descriptors,
    n_signal=config.data.n_signal,
    latent_length=latent_length
)

train_loader, valid_loader = rave_data_loaders(4, dataset, num_workers=8)

# Test
print(next(iter(train_loader)))

"""
Models: (cf Pipeline)
- create models
- train models
- save ckpt and trained model
"""
# instantiate pipeline
pipeline = pipelines[model](config)

# train
if it_ckpt:
    it_ckpt = [int(i) for i in it_ckpt]
"""
seems like pipeline does:
- set model with set_model
- train
Herits "set_data_loader" and "set_data_loader_attr"
"""

pipeline.train(ckpt=ckpt, it_ckpt=it_ckpt)

"""
Real Time:
- load trained model
- eval model
- define z
- pass through decoder y attributes and z
"""
