import os
import matplotlib.pyplot as plt
import sys

import torch
import glob
import numpy as np
import IPython.display as idp
import librosa
from tqdm import tqdm

import torch.nn.functional as F
import torchaudio.transforms as T
import random
import soundfile as sf
from tqdm import tqdm
from raving_fader.helpers.eval import load_model, forward_eval  #, decode_eval
from audio_descriptors.features import compute_all

models = {
    'nsynth': "FRAVE_NSS",
    'darbouka': "FRAVE_DK",
    'japanese': "FRAVE_JPN",
    'violin': "FRAVE_VLN",
    'combined': 'FRAVE_combined',
    'nsynth_full': 'FRAVE_nsynthfull'
}


# Generic function to load a model
def load_rave_model(model_name=models['nsynth']):
    step = 1000000
    path = "/data/ninon/frave_models/"
    ckpt = os.path.join(path + model_name, model_name + "__vae_stage2.ckpt")
    config_file = os.path.join(path + model_name, "train_config.yaml")
    return load_model(config_file, ckpt, datapath=None, batch_size=8)


def get_features_nils(audio, descriptors, sr, latent_length=64):
    feat = compute_all(audio, sr=sr, descriptors=descriptors,
                       mean=False,
                       resample=latent_length)

    feat = {descr: feat[descr] for descr in descriptors}
    feat = np.array(list(feat.values())).astype(np.float32)
    return torch.tensor(feat)


def plot_features(features, descriptors, features_2=None):
    f, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i, ax in enumerate(axs):
        ax.plot(features[i], label="Original")

        if features_2 is not None:
            ax.plot(features_2[i], label="Modified")
        # ax.set_ylim(-1,1)
        ax.set_title(descriptors[i])
        ax.legend()


def normalize(features, all_features):
    features_normed = torch.clone(features)
    for i in range(features.shape[1]):
        features_normed[:, i] = 2 * ((features[:, i] - np.min(all_features[:, i])) / (
                np.max(all_features[:, i]) - np.min(all_features[:, i])) - 0.5)
    return features_normed


def compute(model, audio, features_normed, features_shifted):
    with torch.no_grad():
        x = audio.to(model.device).unsqueeze(0)
        x = model.pqmf(x)
        print(x.shape)
        z, kl = model.reparametrize(*model.encoder(x))
        print(z.shape)
        # z=torch.zeros_like(z)
        z_c = torch.cat((z, features_normed.to(model.device)), dim=1)
        y = model.decoder(z_c, add_noise=False)
        y = model.pqmf.inverse(y)
        y = y.squeeze()

        z_c = torch.cat((z, features_shifted.to(model.device)), dim=1)
        y_mod = model.decoder(z_c, add_noise=False)
        y_mod = model.pqmf.inverse(y_mod)
        y_mod = y_mod.squeeze()

        features_rec = get_features(y_mod.detach().cpu().numpy(), model.descriptors, sr=16000).unsqueeze(0)
        features_rec = normalize(features_rec, all_features)

    return y, z, y_mod, features_rec


def random_eurorack_function(ref_feat):
    n_points = len(ref_feat)
    min, max = torch.min(ref_feat), torch.max(ref_feat)
    linear = np.linspace(-1, 1, n_points)
    linear_inv = np.linspace(1, -1, n_points)
    # Sinusoid generator
    sin_funcs = []
    for i in range(10):
        freq = np.random.randint(1, 30)
        phase = np.random.randn() * 2 * np.pi
        sin_rnd = np.sin((linear * freq) + phase)
        sin_funcs.append(sin_rnd)
    # Square generator
    square_funcs = []
    for f in sin_funcs:
        sq_f = (f > 0) * 2 - 1
    # Sawtooth generator
    sawtooth_funcs = []
    for i in range(10):
        freq = np.random.randint(1, 30)
        phase = np.random.randn() * 2 * np.pi
        sin_rnd = signal.sawtooth(2 * np.pi * freq * linear + phase)
        sawtooth_funcs.append(sin_rnd)
    full_funcs = [[linear, linear_inv], sin_funcs, square_funcs, sawtooth_funcs]
    full_funcs = [item for sublist in full_funcs for item in sublist]
    i = random.randint(0, len(full_funcs) - 1)
    return torch.Tensor(full_funcs[i]) * min


def get_features_phi(signal, pipeline, model, fast_mode=False):
    latent_length = len(signal) // (pipeline.model_config.data_size * np.prod(pipeline.model_config.ratios))
    feat = compute_all(signal, sr=sr, descriptors=model.descriptors,
                       mean=False,
                       resample=latent_length,
                       fast_mode=fast_mode)
    feat = {descr: feat[descr] for descr in model.descriptors}
    feat = np.stack(list(feat.values())).astype(np.float32)
    # Normalize features
    for i, descr in enumerate(model.descriptors):
        feat[i] = pipeline.dataset.normalize(feat[i], pipeline.dataset.min_max_features[descr])
    return torch.tensor(feat)


def get_random_example(name, max_len=10.0):
    files = datasets[name][2]
    sr = datasets[name][1]
    i = random.randint(0, len(files))
    audio, sr = librosa.load(files[i], sr=sr)
    max_len = (max_len * sr) // 2048 * 2048
    audio = audio[:min(max_len, len(audio) // 2048 * 2048)]
    return audio, i


def generate_plot_save(model, audio, feat_in, feat_out, name, sr):
    plot_features(feat_in, feat_out, model.descriptors)
    plt.savefig(name + '.pdf')
    plt.close()
    y = forward_eval(model, torch.tensor(audio).unsqueeze(0), torch.tensor(feat_out).unsqueeze(0)).cpu().numpy()[0]
    sf.write(name + '.wav', y, sr, 'PCM_24')
