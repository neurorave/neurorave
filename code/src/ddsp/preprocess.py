import yaml
import pathlib
import librosa as li
from ddsp.core import extract_loudness, extract_pitch
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
from scipy.io import wavfile
import os
from raving_fader.datasets.attr_dataset import get_dataset_attr
import torchcrepe


def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(x, sampling_rate, block_size, signal_length, oneshot, **kwargs):
    # x, sr = li.load(f, sampling_rate)
    # N = (signal_length - len(x) % signal_length) % signal_length
    # x = np.pad(x, (0, N))

    # if oneshot:
    #     x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    # x = x.reshape(-1, signal_length)
    # pitch = pitch.reshape(x.shape[0], -1)
    # loudness = loudness.reshape(x.shape[0], -1)

    return pitch, loudness


class Dataset(torch.utils.data.Dataset):

    def __init__(self, out_dir):
        super().__init__()
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])
        return s, p, l


class FaderDataset(torch.utils.data.Dataset):

    def __init__(self, out_dir):
        super().__init__()
        # self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))

        features_path = os.path.join("/data/nils/datasets/NSS/",
                                     "features.pth")

        try:
            allfeatures = torch.load(features_path)
        except:
            print('No features file Available')
            allfeatures = None

        self.fader_dataset = get_dataset_attr(
            preprocessed='/data/nils/datasets/NSS',
            wav="/data/nils/datasets/NSS/audio",
            sr=16000,
            descriptors=[
                "centroid", "rms", "bandwidth", "sharpness", "booming"
            ],
            n_signal=65536,
            nb_bins=16,
            latent_length=64,
            r_samples=0.33,
            allfeatures=allfeatures)
        self.pitchs = self.pitchs.reshape(len(self.fader_dataset), -1)
        self.loudness = self.loudness.reshape(len(self.fader_dataset), -1)

    def __len__(self):
        return len(self.fader_dataset)

    def __getitem__(self, idx):
        # s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])

        signal, features = self.fader_dataset[idx]
        signal = torch.from_numpy(signal)
        return signal, p, l, features


def main():

    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    features_path = os.path.join("/data/nils/datasets/NSS/", "features.pth")

    try:
        allfeatures = torch.load(features_path)
    except:
        print('No features file Available')
        allfeatures = None

    dataset = get_dataset_attr(
        preprocessed='/data/nils/datasets/NSS',
        wav="/data/nils/datasets/NSS/audio",
        sr=16000,
        descriptors=["centroid", "rms", "bandwidth", "sharpness", "booming"],
        n_signal=65536,
        nb_bins=16,
        latent_length=64,
        r_samples=0.33,
        allfeatures=allfeatures)

    # files = get_files(**config["data"])
    # pb = tqdm(files)

    # signals = []
    pitchs = []
    loudness = []
    for i in tqdm(range(len(allfeatures))):
        # pb.set_description(str(f))
        signal, _ = dataset[i]
        p, l = preprocess(signal, **config["preprocess"])
        # signals.append(signal)
        pitchs.append(p)
        loudness.append(l)

    # signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    out_dir = '/data/nils/datasets/NSS/ddsp_256'
    makedirs(out_dir, exist_ok=True)

    # np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)


if __name__ == "__main__":
    main()

# def main():
#     class args(Config):
#         CONFIG = "config.yaml"

#     args.parse_args()
#     with open(args.CONFIG, "r") as config:
#         config = yaml.safe_load(config)

#     files = get_files(**config["data"])
#     pb = tqdm(files)

#     signals = []
#     pitchs = []
#     loudness = []

#     for f in pb:
#         pb.set_description(str(f))
#         x, p, l = preprocess(f, **config["preprocess"])
#         signals.append(x)
#         pitchs.append(p)
#         loudness.append(l)

#     signals = np.concatenate(signals, 0).astype(np.float32)
#     pitchs = np.concatenate(pitchs, 0).astype(np.float32)
#     loudness = np.concatenate(loudness, 0).astype(np.float32)

#     out_dir = config["preprocess"]["out_dir"]
#     makedirs(out_dir, exist_ok=True)

#     np.save(path.join(out_dir, "signals.npy"), signals)
#     np.save(path.join(out_dir, "pitchs.npy"), pitchs)
#     np.save(path.join(out_dir, "loudness.npy"), loudness)

# if __name__ == "__main__":
#     main()