import librosa
import numpy as np
import os
from os import path
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from transforms import Reverb, Noise
from utils import init_mel_transform, spectrogram, mel_spectrogram, plot_spectrogram
from collections import defaultdict

import glob
import yaml
from effortless_config import Config

"""
###################
Dataloader for fader on spectrogram representation
Dataset will be composed of spectro and attributes
###################
"""


class Args(Config):
    CONFIG = f"config.yaml"


with open(Args.CONFIG, "r") as config:
    config = yaml.safe_load(config)


class SpectroAttributes(Dataset):
    def __init__(self, root_dir, config):
        # Path directory with wav files
        self.root_dir = root_dir
        # Files names .wav
        self.wav_files = [
                             files_names for files_names in glob.iglob(root_dir + '/**/*.wav', recursive=True)
                             if (files_names.endswith('.wav') or files_names.endswith('.mp3'))
                         ]  # [:64]  # TODO HERRREEEEE
        self.wav_array = np.array(self.wav_files)
        # Path to the spectrogram
        if config["data"]["representation"] == 'spectrogram':
            self.spectro_dir = root_dir + "/spectrograms"
        if config["data"]["representation"] == 'melspectrogram':
            self.spectro_dir = root_dir + "/melspectrograms"
        if not os.path.exists(self.spectro_dir):
            os.mkdir(self.spectro_dir)
            self._spectro_transform(config)
        # Compute spectrogram
        else:
            if config["data"]["export"]:
                self._spectro_transform(config)
        # Files name .pt

        # self.spectro_files = np.array([
        #    files_names for files_names in os.listdir(self.spectro_dir)
        #    if files_names.endswith('.pt')
        # ])
        # Number of tracks in dataset
        self.nb_tracks = np.size(self.wav_array)
        if self.nb_tracks == 0:
            raise Exception("No data found !")

        # Process features
        self.preprocess_function = dummy_load
        self.extension = "*.wav,*.aif"
        self.transforms = None
        self.config = config
        self._preprocess()

        # Create list of indexes
        self.index = np.arange(self.nb_tracks)
        np.random.shuffle(self.index)

    def __len__(self):
        """
        Number of tracks in dataset
        :return: int
        """
        return self.nb_tracks

    def __getitem__(self, index):
        """
        Dataset is composed of the input x [spectrogram or melspectrogram]
        alongside with all the computed attributes by spectral features
        :param index: index of the wav file
        :return: spectre or melspectre, list of features
        """
        spectro = torch.load(self.spectro_files[index]).squeeze()[:, :345]
        features = (self.feats[index] -
                    self.mean_std[0].squeeze()) / self.mean_std[1].squeeze()
        features = torch.tensor(features)  # B x 3
        if config["model"]['n_attr'] == 1:
            features = features[-1].unsqueeze(0)  # B x 1
        # print('Loading file ' + self.spectro_files[index])
        # print(features.shape)
        # print(spectro.shape)
        # dataset = spectro, features
        return spectro, features

    def _spectro_transform(self, config, save_plot=True):
        dict_features = []
        transform = init_mel_transform(config)
        self.spectro_files = []
        for index, wav_index in tqdm(enumerate(self.wav_files),
                                     total=len(self.wav_files)):
            # if os.path.exists(self.spectro_dir + '/impact' + str(index) + '.pt'):
            #    spec = torch.load(self.spectro_dir + '/impact' + str(index) + '.pt')
            #    continue
            # Creation of a spectrogram
            x, sample_rate = torchaudio.load(wav_index)
            # print(sample_rate)
            # print(x.shape)
            # Normalization
            x = x.mean(dim=0)
            shape = config["data"]["max_length"] * sample_rate
            if x.shape[0] > shape:
                m_point = x.shape[0] - shape
                r_point = int(np.random.random() * m_point)
                x_tmp = x[r_point:r_point + int(shape)]
            else:
                x_tmp = torch.cat(
                    [x, torch.zeros(int(shape - int(x.shape[0])))])
            nb_rep = 0
            # Padding of 0 to have equal shape for tensors
            if x.shape[0] < shape:
                x_tmp = torch.cat(
                    [x, torch.zeros(int(shape - int(x.shape[0])))])
            while torch.mean(torch.abs(x_tmp)) < 1e-4:
                m_point = x.shape[0] - shape
                r_point = int(np.random.random() * m_point)
                x_tmp = x[r_point:r_point + int(shape)]
                nb_rep += 1
                if nb_rep > 3:
                    break
            x = x_tmp.unsqueeze(0)
            # print('Prior to spectro compute / Signal len')
            # print(x.shape)
            # Compute Representation
            if config["data"]["representation"] == 'spectrogram':
                spec = spectrogram(x, config)
                spec = torch.log10(spec + 0.1)
            if config["data"]["representation"] == 'melspectrogram':
                spec = mel_spectrogram(x, transform, config)
                spec = torch.log10(spec + 0.1)
                # print("spec shape")
                # print(spec.shape)
                # [1 * 80 * 32]
            if save_plot:
                plot_spectrogram(spec[0],
                                 self.spectro_dir,
                                 title=config["data"]["dataset"] + str(index))
                plt.close()
            # Save the spectrogram
            # print('Saving this spectro shape')
            # print(spec.shape)
            torch.save(
                spec, self.spectro_dir + '/' + config["data"]["dataset"] +
                      str(index) + '.pt')
            # print(self.spectro_dir + '/' + config["data"]["dataset"] + str(index) + '.pt')
            self.spectro_files.append(self.spectro_dir + '/' + config["data"]["dataset"] + str(index) + '.pt')
            del spec
            del x
        # Gather features values & print histogram of features for data check
        # new_dic = defaultdict(list)
        # for d in dict_features:
        #     for k, v in d.items():
        #         new_dic[k].append(v)
        # for key, values in new_dic.items():
        #     plt.figure()
        #     plt.hist(values, density=False, bins=100,
        #              color='red')  # density=False would make counts
        #     plt.title('Features distribution')
        #     plt.ylabel('Distribution')
        #     plt.xlabel(key)
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.savefig(config["data"]["dataset_features_path"] + '/' +
        #                 str(key) + ".pdf")
        #     plt.close()

    def _preprocess(self):
        extension = self.extension.split(",")
        idx = 0
        wavs = []
        # Populate the file list
        if self.root_dir is not None:
            for f, folder in enumerate(self.root_dir.split(",")):
                print("Recursive search in {}".format(folder) +
                      " for features extraction")
                for ext in extension:
                    wavs.extend(list(Path(folder).rglob(ext)))
                    wavs = wavs  # [:64]  # TODO HERRREEEEE
        # else:
        #     with open(self.file_list, "r") as file_list:
        #         wavs = file_list.read().split("\n")
        self.env = [None] * len(wavs)
        self.feats = [None] * len(wavs)
        loader = tqdm(wavs)
        for wav in loader:
            loader.set_description("{}".format(path.basename(wav)))
            output = self.preprocess_function(wav, self.config)
            # print(output.shape)
            output = output / (np.max(np.abs(output)) + 1e-9)
            # print(output.shape)
            # output = output - np.mean(output)
            feats = spectral_features_spectro(output, config["preprocess"]["sample_rate"])
            # print(output.shape)
            feats = np.mean(feats, axis=1)
            # print(output.shape)
            if idx > 0:
                output = np.zeros((1, 1))
            if output is not None:
                self.env[idx] = output
                self.feats[idx] = feats
                idx += 1
            # print("output")
            # print(output.shape)
        self.env = np.array(self.env)
        self.feats = np.array(self.feats)
        self.mean_std = [None] * 2
        self.mean_std[0] = np.mean(self.feats, axis=0).astype(float)
        self.mean_std[1] = np.max(self.feats, axis=0).astype(float)
        self.len = len(self.env)


def dummy_load(name, config):
    """
    Preprocess function that takes one audio path and load it into
    chunks of 2048 samples.
    """
    x = librosa.load(str(name), config["preprocess"]["sample_rate"])[0]
    # shape = args.max_length * args.sample_rate
    if x.shape[0] > config["data"]["shape"]:
        m_point = x.shape[0] - config["data"]["shape"]
        r_point = int(np.random.random() * m_point)
        x = x[r_point:r_point + int(config["data"]["shape"])]
    else:
        x = np.concatenate(
            [x, np.zeros(int(config["data"]["shape"] - int(x.shape[0])))])
    return x


def spectral_features_spectro(S, sr):
    features = [None] * 4
    # Spectral features
    # S, phase = librosa.magphase(librosa.stft(y=y))
    # Compute all descriptors
    features[0] = np.mean(S, axis=0)[np.newaxis, :]
    features[1] = librosa.feature.spectral_rolloff(S=S)
    features[2] = librosa.feature.spectral_bandwidth(S=S)
    features[3] = librosa.feature.spectral_centroid(S=S)
    # print(features[6].shape)
    features = np.concatenate(features)
    features[np.isnan(features)] = 1
    features = features[:, :-1]
    return features


def spectral_features(y, sr):
    features = [None] * 7
    features[0] = librosa.feature.rms(y)
    features[1] = librosa.feature.zero_crossing_rate(y)
    # Spectral features
    S, phase = librosa.magphase(librosa.stft(y=y))
    # Compute all descriptors
    features[2] = librosa.feature.spectral_rolloff(S=S)
    features[3] = librosa.feature.spectral_flatness(S=S)
    features[4] = librosa.feature.spectral_bandwidth(S=S)
    features[5] = librosa.feature.spectral_centroid(S=S)
    features[6] = librosa.yin(y, 50, 5000, sr=sr)[np.newaxis, :]
    # print(features[6].shape)
    features = np.concatenate(features)
    features[np.isnan(features)] = 1
    features = features[:, :-1]
    return features


def import_fader_dataset(config):
    # Final dataset directory
    final_dir = config["data"]["data_location"] + '/' + config["data"][
        "dataset"]
    # Compute shape
    config["data"]["shape"] = np.ceil(
        (config["data"]["duration"] * config["preprocess"]["sample_rate"] /
         512.)) * 512
    # Create a set of transforms
    if config["data"]["transform"] and config["model"]["model"] == "vae":
        transform = transforms.RandomApply([
            transforms.RandomChoice([
                Noise(1e-5),
                Reverb(0.001, 0.01, config["preprocess"]["sample_rate"]),
            ])
        ], 0.3)
        config["data"]["transform"] = transform
    else:
        config["data"]["transform"] = None, None
    print("Spectrogram processing")
    dataset = SpectroAttributes(final_dir, config)
    # if not os.path.exists(config["data"]["loaders_path"] + '.th'):
    #     try:
    #         os.makedirs(config["data"]["loaders_path"])
    #     except:
    #         pass
    dataset_size = len(dataset)
    print("Dataset Size: " + str(dataset_size))
    # compute indices for train/test split
    indices = np.array(list(range(dataset_size)))
    split = int(np.floor(config["data"]["test_size"] * dataset_size))
    if config["data"]["shuffle_dataset"]:
        np.random.seed(config["train"]["seed"])
        np.random.shuffle(indices)
    global_train_indices, test_idx = np.array(indices[split:]), np.array(
        indices[:split])
    # Compute indices
    split = int(
        np.floor(config["data"]["valid_size"] * len(global_train_indices)))
    # Shuffle examples
    np.random.shuffle(global_train_indices)
    # Split the train set to obtain a validation set
    train_idx, valid_idx = global_train_indices[
                           split:], global_train_indices[:split]
    # if mini quick train needed
    if config["data"]["subsample"] > 0:
        train_idx = train_idx[:config["data"]["subsample"]]
        valid_idx = valid_idx[:config["data"]["subsample"]]
        test_idx = test_idx[:config["data"]["subsample"]]
    # Create corresponding subsets
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    # Create all the spectro loaders
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["nb_workers"],
        drop_last=True,
        sampler=train_sampler,
        pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["nb_workers"],
        drop_last=True,
        sampler=valid_sampler,
        pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["nb_workers"],
        drop_last=True,
        sampler=test_sampler,
        shuffle=False,
        pin_memory=False)
    if config["debug"]:
        for x in range(len(dataset)):
            print("Shape of spectro:")
            print(dataset[x][0].shape)
            print("Shape of features")
            print(dataset[x][1].shape)
        batch_x, batch_y = next(iter(train_loader))
        config["data"]["input_size"] = batch_x.shape
        print(config["data"]["input_size"])
        config["data"]["attribute_size"] = batch_y.shape
        print(config["data"]["attribute_size"])
    # Save as a torch object
    torch.save([train_loader, valid_loader, test_loader],
               config["data"]["loaders_path"] + '.th')
    return train_loader, valid_loader, test_loader, config


class RandomDataset(Dataset):
    def __init__(self, config):
        self.nb_tracks = 3600
        self.index = np.arange(self.nb_tracks)

    def __len__(self):
        """
        Number of tracks in random dataset
        :return: int
        """
        return self.nb_tracks

    def __getitem__(self, index):
        """
        Dataset is composed of random input x [size spectrogram or melspectrogram]
        alongside with all the random attributes
        :param index: index of the wav file
        :return: spectre or melspectre, list of features
        """
        spectro = torch.rand(80, 87, device=config["train"]["device"])
        features = torch.rand(7, device=config["train"]["device"])
        return spectro, features


def import_random(config):
    dataset = RandomDataset(config)
    dataset_size = len(dataset)
    print("Dataset Size: " + str(dataset_size))
    # compute indices for train/test split
    indices = np.array(list(range(dataset_size)))
    split = int(np.floor(config["data"]["test_size"] * dataset_size))
    if config["data"]["shuffle_dataset"]:
        np.random.seed(config["train"]["seed"])
        np.random.shuffle(indices)
    global_train_indices, test_idx = np.array(indices[split:]), np.array(
        indices[:split])
    # Compute indices
    split = int(
        np.floor(config["data"]["valid_size"] * len(global_train_indices)))
    # Shuffle examples
    np.random.shuffle(global_train_indices)
    # Split the train set to obtain a validation set
    train_idx, valid_idx = global_train_indices[
                           split:], global_train_indices[:split]
    # if mini quick train needed
    if config["data"]["subsample"] > 0:
        train_idx = train_idx[:config["data"]["subsample"]]
        valid_idx = valid_idx[:config["data"]["subsample"]]
        test_idx = test_idx[:config["data"]["subsample"]]
    # Create corresponding subsets
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    # Create all the spectro loaders
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["nb_workers"],
        drop_last=True,
        sampler=train_sampler,
        pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["nb_workers"],
        drop_last=True,
        sampler=valid_sampler,
        pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["nb_workers"],
        drop_last=True,
        sampler=test_sampler,
        shuffle=False,
        pin_memory=False)
    return train_loader, valid_loader, test_loader
