import torch
# from . import SimpleLMDBDataset
from pathlib import Path
import librosa as li
from os import makedirs, path
from tqdm import tqdm
import numpy as np
from scipy.io.wavfile import read as read_wav_file
from audio_descriptors.features import compute_all
from udls.transforms import Compose, RandomApply, Dequantize, RandomCrop
from raving_fader.datasets.rave_dataset import random_phase_mangle
from udls import SimpleLMDBDataset
import torchaudio.transforms as T


def dummy_load(name):
    """
    Preprocess function that takes one audio path and load it into
    chunks of 2048 samples.
    """
    x = li.load(name, 16000)[0]
    if len(x) % 2048:
        x = x[:-(len(x) % 2048)]
    x = x.reshape(-1, 2048)
    if x.shape[0]:
        return x
    else:
        return None


def simple_audio_preprocess(sampling_rate, N):

    def preprocess(name):
        try:
            x, sr = li.load(name, sr=sampling_rate)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            return None

        pad = (N - (len(x) % N)) % N
        x = np.pad(x, (0, pad))

        x = x.reshape(-1, N)
        return x.astype(np.float32)

    return preprocess


class SimpleDatasetAttr(torch.utils.data.Dataset):

    def __init__(self,
                 out_database_location,
                 folder_list=None,
                 file_list=None,
                 preprocess_function=dummy_load,
                 transforms=None,
                 extension="*.wav,*.aif",
                 map_size=1e12,
                 split_percent=.2,
                 split_set="full",
                 seed=0,
                 descriptors=[],
                 sr=16000,
                 latent_length=64,
                 nb_bins=16,
                 r_samples=None,
                 allfeatures=None):
        super().__init__()

        assert folder_list is not None or file_list is not None
        makedirs(out_database_location, exist_ok=True)
        self.env = SimpleLMDBDataset(out_database_location, map_size)

        self.folder_list = folder_list
        self.file_list = file_list

        self.preprocess_function = preprocess_function
        self.extension = extension

        self.transforms = transforms

        self.descriptors = descriptors
        self.latent_length = latent_length
        self.sr = sr
        # self.min_max_features=min_max_features
        # self.mean_var_features=mean_var_features
        self.allfeatures = allfeatures

        # IF NO DATA INSIDE DATASET: PREPROCESS
        self.len = len(self.env)

        if self.len == 0:
            self._preprocess()
            self.len = len(self.env)
        print(self.len)
        print("begin here")
        if self.len == 0:
            raise Exception("No data found !")

        self.r_samples = r_samples

        if self.allfeatures is None:
            self.compute_min_max()

        self.min_max_features = {}
        for i, descr in enumerate(self.descriptors):
            self.min_max_features[descr] = [
                np.min(self.allfeatures[:, i, :]),
                np.max(self.allfeatures[:, i, :])
            ]

        self.compute_bins(nb_bins)

        self.index = np.arange(self.len)
        np.random.seed(seed)
        np.random.shuffle(self.index)

        if split_set == "train":
            self.len = int(np.floor((1 - split_percent) * self.len))
            self.offset = 0

        elif split_set == "test":
            self.offset = int(np.floor((1 - split_percent) * self.len))
            self.len = self.len - self.offset

        elif split_set == "full":
            self.offset = 0

    def _preprocess(self):
        extension = self.extension.split(",")
        idx = 0
        wavs = []

        # POPULATE WAV LIST
        if self.folder_list is not None:
            for f, folder in enumerate(self.folder_list.split(",")):
                print("Recursive search in {}".format(folder))
                for ext in extension:
                    wavs.extend(list(Path(folder).rglob(ext)))
            wavs = [str(w) for w in wavs]
        else:
            with open(self.file_list, "r") as file_list:
                wavs = file_list.read().split("\n")

        print(len(wavs), " files found")
        loader = tqdm(wavs)
        for wav in loader:
            loader.set_description("{}".format(path.basename(wav)))
            output = self.preprocess_function(wav)
            if output is not None:
                for o in output:
                    # tresh = 0.001
                    # above = o[np.abs(o) > tresh]
                    # ratio = len(above) / len(o)
                    # if ratio > 0.4:
                    self.env[idx] = o
                    idx += 1

    def compute_min_max(self):
        indices = range(self.len)
        self.allfeatures = []
        if self.sr < 44100:
            resampler = T.Resample(self.sr, 44100)
        for i in tqdm(indices):
            try:

                features = compute_all(self.env[i],
                                       sr=self.sr,
                                       descriptors=self.descriptors,
                                       mean=False,
                                       resample=self.latent_length,
                                       resampler=resampler)

                features = {
                    descr: features[descr]
                    for descr in self.descriptors
                }
                feature_arr = np.array(list(features.values())).astype(
                    np.float32)

                self.allfeatures.append(feature_arr)
                first = True
            except:
                # break
                print('failed features compute')
                print('index failure : ', i)
                self.allfeatures.append(
                    np.zeros((len(self.descriptors),
                              self.latent_length)).astype(np.float32))

        self.allfeatures = np.array(self.allfeatures)

    def compute_bins(self, nb_bins):
        all_values = []
        for i, descr in enumerate(self.descriptors):
            data = self.allfeatures[:, i, :].flatten()
            data[data < 0] = 0
            data_true = data.copy()
            data.sort()
            index = np.linspace(0, len(data) - 2, nb_bins).astype(int)
            values = [data[i] for i in index]
            all_values.append(values)
        self.bin_values = np.array(all_values)

    def __len__(self):
        return self.len

    def normalize(self, array, min_max):
        return 2 * ((array - min_max[0]) / (min_max[1] - min_max[0]) - 0.5)

    def unnormalize(self, array, min_max):
        array = (array + 0.5) / 2
        return array * (min_max[1] - min_max[0]) + min_max[0]

    def normalize_all(self, attr):
        if len(attr.shape) == 3:
            for i, descr in enumerate(self.descriptors):
                attr[:, i] = self.normalize(attr[:, i],
                                            self.min_max_features[descr])
        else:
            for i, descr in enumerate(self.descriptors):
                attr[i] = self.normalize(attr[i], self.min_max_features[descr])
        return attr

    def __getitem__(self, index):
        data = self.env[index]
        if self.transforms is not None:
            data = self.transforms(data)
        feature_arr = torch.tensor(self.allfeatures[index].copy())
        return data, feature_arr


def get_dataset_attr(preprocessed,
                     wav,
                     sr,
                     n_signal,
                     descriptors,
                     latent_length,
                     nb_bins,
                     allfeatures=None,
                     r_samples=None):
    dataset = SimpleDatasetAttr(
        preprocessed,
        wav,
        preprocess_function=simple_audio_preprocess(sr, n_signal),
        transforms=Compose([
            RandomApply(
                lambda x: random_phase_mangle(x, 20, 2000, .99, sr),
                p=.8,
            ),
            Dequantize(16),
            lambda x: x.astype(np.float32),
        ]),
        descriptors=descriptors,
        sr=sr,
        latent_length=latent_length,
        allfeatures=allfeatures,
        r_samples=r_samples)
    return dataset
