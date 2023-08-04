from ddsp.model import DDSP
from raving_fader.datasets.data_loaders import rave_data_loaders
from preprocess import FaderDataset
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness

from os import path
import yaml
import torch
import torchaudio

run = "/data/nils/raving-fader/src/ddsp/fddsp"

####### Import the model #######

with open(path.join(run, "config.yaml"), "r") as config:
    config = yaml.safe_load(config)

ddsp = DDSP(**config["model"])

state = ddsp.state_dict()
pretrained = torch.load(path.join(run, "state.pth"), map_location="cpu")
state.update(pretrained)
ddsp.load_state_dict(state)

name = path.basename(path.normpath(run))

####### Import the dataset #######

out_dir = config["preprocess"]["out_dir"]
dataset = FaderDataset(out_dir)
dataloader, validloader = rave_data_loaders(16, dataset, num_workers=8)

mean_loudness, std_loudness = mean_std_loudness(
    torch.from_numpy(dataset.loudness))
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

####### MFCC #######

wave2mfcc = torchaudio.transforms.MFCC(sample_rate=16000,
                                       n_mfcc=30,
                                       melkwargs={
                                           "n_fft":
                                           1024,
                                           "hop_length":
                                           config["preprocess"]["block_size"],
                                           "f_min":
                                           20,
                                           "f_max":
                                           int(16000 / 2),
                                           "n_mels":
                                           128,
                                           "center":
                                           True
                                       })

####### Test Inference #######

stot, p, l, features = next(iter(dataloader))
s, p, l, features = stot[:1], p[:1].unsqueeze(-1), l[:1].unsqueeze(
    -1), features[:1]

l = (l - mean_loudness) / std_loudness

mfcc = wave2mfcc(s)[:, :, :-1]
mfcc = mfcc.permute(0, 2, 1)

s_out = s

for signal in stot:
    signal = signal.unsqueeze(0)
    mfcc = wave2mfcc(signal)[:, :, :-1]
    mfcc = mfcc.permute(0, 2, 1)
    z = ddsp.encoder(mfcc)
    s_rec = ddsp.decode(z, p, l).squeeze(-1)
    s_out = torch.cat((s_out, s_rec), dim=-1)

outpath = "/data/nils/raving-fader/src/ddsp/test"
torchaudio.save(path.join(outpath, "test_inference.wav"),
                s_out,
                sample_rate=16000)
