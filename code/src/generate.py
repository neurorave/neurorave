import os
import sys
from utils import load_rave_model, plot_features, normalize, \
    compute, random_eurorack_function, generate_plot_save

import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as idp
import librosa
from tqdm import tqdm

import torch.nn.functional as F
import torchaudio.transforms as T
import random
import soundfile as sf
from tqdm import tqdm

from raving_fader.helpers.helper_plot import hdr_plot_style  # , plot_features


# os.chdir("/data/ninon/raving-fader/src")

from audio_descriptors.features import compute_all


# from raving_fader.helpers.eval import load_model

def get_random_example(name, max_len=10.0):
    files = datasets[name][2]
    sr = datasets[name][1]
    i = random.randint(0, len(files))
    audio, sr = librosa.load(files[i], sr=sr)
    max_len = (max_len * sr) // 2048 * 2048
    print(max_len)
    audio = audio[:min(max_len, len(audio) // 2048 * 2048)]
    return audio, i


def get_features(signal, pipeline, model):
    latent_length = len(signal) // (pipeline.model_config.data_size * np.prod(pipeline.model_config.ratios))
    feat = compute_all(signal, sr=sr, descriptors=model.descriptors,
                       mean=False,
                       resample=latent_length)
    feat = {descr: feat[descr] for descr in model.descriptors}
    feat = np.stack(list(feat.values())).astype(np.float32)
    # Normalize features
    for i, descr in enumerate(model.descriptors):
        feat[i] = pipeline.dataset.normalize(feat[i], pipeline.dataset.min_max_features[descr])
    return torch.tensor(feat)


torch.set_grad_enabled(False)

models_dir = "/data/ninon/raving-fader/models/"
model_name = "FRAVE_nsynthfull"  # /CRAVE_nsynthfull
step = 1000000
datapath = "/data/ninon/datasets/nsynth_full_valid"

model_name = "FRAVE_combined"  # /CRAVE_combined
datapath = "/data/ninon/datasets/combined"

'''
HERE FIND ALL THE MISSING MODELS (ALL EXCEPT COMBINED/FULL)
'''
models_path = "/data/ninon/frave_models/"
models = {
    # 'nsynth': "FRAVE_NSS",
    # 'darbouka': "FRAVE_DK",
    # 'japanese': "FRAVE_JPN",
    'violin': "FRAVE_VLN",
    # 'combined': 'FRAVE_combined',
    # 'nsynth_full': 'FRAVE_nsynthfull'
}

datasets = {
    # 'nsynth': ("/data/nils/datasets/NSS/audio", 16000),
    # 'japanese': ("/data/nils/datasets/JPN/audio/", 48000),
    # 'darbouka': ("/data/nils/datasets/DK/audio/", 48000),
    'violin': ("/data/nils/datasets/VLN/audio/", 44100),
    # 'nsynth_full': ("/data/nils/datasets/nsynth_full_valid/audio/", 16000),
    # 'combined': ("/data/nils/datasets/combined/audio", 16000)
}

# Retrieve information
# config, pipeline, model, checkpoint = load_rave_model('nsynth_full')
# trainloader = pipeline.train_set
# all_features = pipeline.dataset.allfeatures
# Retrieve all informations
config, pipeline, model, checkpoint = load_rave_model()
# Most important config information
pipeline.val_set
pipeline.latent_length
pipeline.data_config.sr
pipeline.data_config.n_signal
pipeline.data_config.descriptors

print("step:")
print(checkpoint["step"])

audio = pipeline.dataset.env[0]

# Recursive file seeking
for name, (path, sr) in datasets.items():
    files = []
    # Fetch all test wav files
    for f in glob.glob(path + '/**/*.wav', recursive=True):
        files.append(f)
    # Retrieve the current validation set
    config, pipeline, model, checkpoint = load_rave_model(model_name=models[name])
    val_files = []
    for x, feats in pipeline.val_set:
        for b in range(x.shape[0]):
            val_files.append([x[b], feats[b]])
    del pipeline.val_set
    datasets[name] += (files, val_files,)
# Check one random sample per dataset
for name, (path, sr, files, val_set) in datasets.items():
    print(f"Dataset {name} - {len(files)} examples - {len(val_set)} validation")
    i = random.randint(0, len(files))

'''
Only try to improve the violin problem
'''
base_path = '/data/ninon/output/transfer'
# Set of sources
for model_test in ['violin']:
    # Import model
    config, pipeline, model, checkpoint = load_rave_model(models[model_test])
    for audio_test in ['violin']:
        for feat_test in ['violin']:
            for i in range(3):
                # Retrieve random audio base
                print("retrieve random audio base")
                audio, a_i = get_random_example(audio_test)
                latent_length = len(audio) // (pipeline.model_config.data_size * np.prod(pipeline.model_config.ratios))
                feat_in = get_features(audio, pipeline, model)
                # Retrieve random feature base
                audio_f, f_i = get_random_example(feat_test)
                audio_feat = np.zeros(len(audio))
                audio_feat[:min(len(audio), len(audio_f))] = audio_f[:len(audio)]
                feat_change = get_features(audio_feat, pipeline, model)
                base_name = base_path + '/' + model_test + '_' + audio_test + '_' + feat_test + '_' + str(
                    i) + '_' + str(a_i) + '_' + str(f_i)
                sf.write(base_name + '_audio.wav', audio, sr, 'PCM_24')
                sf.write(base_name + '_feats.wav', audio_feat, sr, 'PCM_24')
                feat_cumulate = feat_in.clone().detach()
                feat_cumulate_mix = feat_in.clone().detach()
                for j in [4, 3, 2, 1, 0]:
                    # Switch one feature
                    print("Switch one feature")
                    feat_switch = feat_in.clone().detach()
                    feat_switch[j] = feat_change[j]
                    generate_plot_save(model, audio, feat_in, feat_switch, base_name + '_change_' + str(j), sr)
                    # Mix features
                    feat_add = feat_in.clone().detach()
                    feat_add[j] = feat_add[j] + feat_change[j]
                    generate_plot_save(model, audio, feat_in, feat_add, base_name + '_mix_' + str(j), sr)
                    # Eurorack features
                    eurorack_func = random_eurorack_function(feat_in[j])
                    feat_euro = feat_in.clone().detach()
                    feat_euro[j] = eurorack_func
                    generate_plot_save(model, audio, feat_in, feat_euro, base_name + '_eurorack_' + str(j), sr)
                    # Eurorack features
                    eurorack_func = random_eurorack_function(feat_in[j])
                    feat_euro_add = feat_in.clone().detach()
                    feat_euro_add[j] = feat_euro_add[j] + eurorack_func
                    generate_plot_save(model, audio, feat_in, feat_euro_add, base_name + '_eurorack_mix_' + str(j), sr)
                    # Cumulative mix features
                    feat_cumulate[j] = feat_change[j]
                    generate_plot_save(model, audio, feat_in, feat_cumulate, base_name + '_cumulate_' + str(j), sr)
                    feat_cumulate_mix[j] = feat_cumulate_mix[j] + feat_change[j]
                    generate_plot_save(model, audio, feat_in, feat_cumulate_mix, base_name + '_cumulate_mix_' + str(j),
                                       sr)

# # From a wav file
# wav_path = "/data/nils/datasets/nsynth_full_valid/audio/"
# id_wav = random.choice(range(len(os.listdir(wav_path))))
# wav_file = os.path.join(wav_path, os.listdir(wav_path)[id_wav])
# audio, _ = librosa.load(wav_file, sr=model.sr)
# features = get_features(audio, model.descriptors, sr=model.sr).unsqueeze(0)
# N = 65536
# pad = (N - (len(audio) % N)) % N
# audio = np.pad(audio, (0, pad))
# audio = torch.tensor(audio).unsqueeze(0).to(model.device)
#
# # From the dataloader
# audio, _ = next(iter(trainloader))
# audio = audio[:1]
# features = get_features(audio.numpy().squeeze(), model.descriptors, sr=model.sr).unsqueeze(0)
#
# # Compute reconstruction
# # Descriptors
# # Use the orignal descriptors of the sound for reconstruction
# features_normed = normalize(features, all_features)
# features_shifted = torch.clone(features_normed)
# # Change the descriptors. For instance here: shift all the descriptors of a given offset
# descr = 0
# offset = 0.2
#
# features_normed = normalize(features, all_features)
# features_shifted = torch.clone(features_normed)
# features_shifted[:, descr] = features_shifted[:, descr] + offset
#
# # Reconstruction and results
# y, z, y_mod, features_rec = compute(model, audio, features_normed, features_shifted)
# f, axs = plt.subplots(1, 5, figsize=(20, 5))
# for i, ax in enumerate(axs):
#     ax.plot(features_normed.squeeze()[i], label="Original")
#     ax.plot(features_shifted.squeeze()[i], label="Target")
#     ax.plot(features_rec.squeeze()[i], label="Reconstructed")
#     # ax.set_ylim(-1,1)
#     ax.set_title(model.descriptors[i])
#     ax.legend()
#
# print("Original Audio")
# idp.display(idp.Audio(audio.cpu().numpy(), rate=model.sr))
#
# print("Reconstruction with same attributes")
# idp.display(idp.Audio(y.cpu().numpy(), rate=model.sr))
#
# print("Reconstruction with modified attributes")
# idp.display(idp.Audio(y_mod.cpu().numpy(), rate=model.sr))
