from preprocess import FaderDataset
import os
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import torch
from raving_fader.datasets.data_loaders import rave_data_loaders

out_dir = '/data/nils/datasets/NSS'

dataset = FaderDataset(out_dir)

trainloader, validloader = rave_data_loaders(32, dataset, num_workers=8)

signal, p, l, features = next(iter(validloader))
signal, p, l, features = signal[0], p[0], l[0], features[0]

outpath = "/data/nils/raving-fader/src/ddsp/test"

torchaudio.save(os.path.join(outpath, "test.wav"),
                signal.unsqueeze(0),
                sample_rate=16000)

f, ax = plt.subplots(4, 2)
ax = ax.flatten()

ax[0].plot(p)
ax[0].set_title("pitch")

ax[1].plot(l)
ax[1].set_title("loudness")

descriptors = dataset.fader_dataset.descriptors
for i in range(len(descriptors)):
    ax[i + 2].plot(features[i])
    ax[i + 2].set_title(descriptors[i])
f.savefig(os.path.join(outpath, "features.png"))
