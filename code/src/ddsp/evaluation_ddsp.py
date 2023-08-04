import os
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm
import numpy as np

from ddsp.model import DDSP
from raving_fader.datasets.data_loaders import rave_data_loaders
from raving_fader.models.rave.core import multiscale_stft
from preprocess import FaderDataset
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness

from os import path
import yaml
import torchaudio
import cdpam

from raving_fader.helpers.evaluation_functions import load_model, get_features, normalize, swap_features
from effortless_config import Config

import matplotlib.pyplot as plt

torch.set_grad_enabled(False)


def get_mel_loss(x, y, nfft=2048, nmels=512, hop_length=512, win_length=2048):
    transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=nfft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        # norm='slaney',
        onesided=True,
        n_mels=nmels,
    ).to(device)
    X_mel = transform(x)
    Y_mel = transform(y)
    loss_melspectro = F.l1_loss(X_mel, Y_mel)
    return loss_melspectro


def get_jnd_loss(x, y, loss_fn):
    x, y = x.squeeze(), y.squeeze()
    with torch.no_grad():
        resample_rate = 22050
        resampler = T.Resample(16000, resample_rate, dtype=y.dtype).to(device)
        jnd_loss = torch.mean(
            loss_fn.forward(resampler(x) * 32768,
                            resampler(y) * 32768)).item()
    return jnd_loss


def lin_distance(x, y):
    return torch.norm(x - y) / torch.norm(x)


def log_distance(x, y):
    return abs(torch.log(x + 1e-7) - torch.log(y + 1e-7)).mean()


def distance(x, y):
    scales = [2048, 1024, 512, 256, 128]
    x = multiscale_stft(x, scales, .75)
    y = multiscale_stft(y, scales, .75)

    lin = sum(list(map(lin_distance, x, y)))
    log = sum(list(map(log_distance, x, y)))

    return lin + log


### CONFIG ###
class args(Config):
    RUN = None


args.parse_args()

# Params
run = "./runs/FDDSP"

nb_swap = 5  # Number of attributes to swap

if args.RUN is not None:
    run = args.RUN

## Import the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(path.join(run, "config.yaml"), "r") as config:
    config = yaml.safe_load(config)

model = DDSP(**config["model"]).to(device)
model.eval()

state = model.state_dict()
pretrained = torch.load(path.join(run, "state.pth"), map_location="cpu")
state.update(pretrained)
model.load_state_dict(state)

name = path.basename(path.normpath(run))
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
                                       }).to(device)

## Get the validation set dataloader
out_dir = config["preprocess"]["out_dir"]
dataset = FaderDataset(out_dir)

#WARNIIIING TO CHANGE #############################################################################################
dataloadeer, validloader = rave_data_loaders(16, dataset, num_workers=8)

mean_loudness, std_loudness = mean_std_loudness(
    torch.from_numpy(dataset.loudness))
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

# Import the features from the full dataset
# all_features = torch.load(os.path.join(dataset_dir, "features.pth"))
all_features = dataset.fader_dataset.allfeatures
##########  COMPUTATION LOOP ############

#Create the results dict
results = {
    "distance": 0,
    "mel": 0,
    "jnd": 0,
    "correlation": {descr: 0
                    for descr in dataset.fader_dataset.descriptors},
    "l1_loss": {descr: 0
                for descr in dataset.fader_dataset.descriptors}
}

# Resampler for timbral models descriptors
resampler = T.Resample(16000, 44100).to(device)  #
loss_fn = cdpam.CDPAM(dev=device)

first = True
# Iterate over the validation set
for k, (s, p, l, features) in enumerate(tqdm(validloader)):

    # Get x and features from dataloader
    features = features.to(device)
    #Normalize the features
    features_normed = normalize(features, all_features)
    features_normed = features_normed

    #Create the swapped features
    features_swapped = swap_features(all_features, features_normed, nb_swap)

    #Reconstruct the signals
    s = s.to(device)
    mfcc = wave2mfcc(s)[:, :, :-1]
    mfcc = mfcc.permute(0, 2, 1)
    p = p.unsqueeze(-1).to(device)
    l = l.unsqueeze(-1).to(device)
    l = (l - mean_loudness) / std_loudness

    z = model.encoder(mfcc)

    z_c_normal = torch.cat((z, torch.permute(features_normed, (0, 2, 1))),
                           dim=2)
    z_c_swapped = torch.cat((z, torch.permute(features_swapped, (0, 2, 1))),
                            dim=2)

    y = model.decode(z_c_normal, p, l).squeeze(-1)

    s, y = s.unsqueeze(1), y.unsqueeze(1)

    loss = distance(s, y).item()
    mel_loss = get_mel_loss(s, y).item()
    jnd = get_jnd_loss(s, y, loss_fn)

    results["distance"] += loss / len(validloader)
    results["jnd"] += jnd / len(validloader)
    results["mel"] += mel_loss / len(validloader)

    # y_mod = model.decode(z_c_swapped, p, l).squeeze(-1)

    # #Resample using torchaudio for faster feature computation

    # #Recompute the features
    # features_rec = torch.zeros_like(features_normed)
    # # if the model sample rate is below 44100, resample the signal with torchaudio for faster computation of timnbra descriptors
    # y_mod_resamp = resampler(y_mod)

    # for i, (signal, signal_resamp) in enumerate(
    #         zip(y_mod.squeeze().detach().cpu().numpy(),
    #             y_mod_resamp.squeeze().detach().cpu().numpy())):
    #     features_rec[i, :3] = get_features(
    #         signal, dataset.fader_dataset.descriptors[:3],
    #         sr=16000)  #Using the raw signal for librosa descriptors
    #     features_rec[i, 3:] = get_features(
    #         signal_resamp, dataset.fader_dataset.descriptors[3:], sr=44100
    #     )  #Using the torchaudio-resampled signal for timbra models (faster)

    # features_rec = normalize(features_rec, all_features)

    # #Compute the losses
    # # Loss rec
    # ori_stft = multiscale_fft(
    #     s,
    #     config["train"]["scales"],
    #     config["train"]["overlap"],
    # )
    # rec_stft = multiscale_fft(
    #     y,
    #     config["train"]["scales"],
    #     config["train"]["overlap"],
    # )

    # loss = 0
    # for s_x, s_y in zip(ori_stft, rec_stft):
    #     lin_loss = (s_x - s_y).abs().mean()
    #     log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
    #     loss = loss + lin_loss + log_loss

    # results["distance"] += loss.item() / len(validloader)

    # # Loss DESCR
    # for i, descr in enumerate(dataset.fader_dataset.descriptors):
    #     feat_out = features_rec[:, i]
    #     feat_in = features_swapped[:, i]

    #     l1_loss_descr = F.l1_loss(feat_in, feat_out).item()
    #     results["l1_loss"][descr] += l1_loss_descr / len(validloader)

    #     cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    #     correlation = (((cos(
    #         feat_in.reshape(-1) - feat_in.mean(),
    #         feat_out.reshape(-1) - feat_out.mean()) + 1) / 2)).item()
    #     results["correlation"][descr] += correlation / len(validloader)

    # if first == True:
    #     f, axs = plt.subplots(16, 5, figsize=(20, 80))
    #     features_swapped = features_swapped.cpu()
    #     features_normed = features_normed.cpu()
    #     features_rec = features_rec.cpu()
    #     for j in range(16):
    #         axcur = axs[j]

    #         for i, ax in enumerate(axcur):
    #             ax.plot(features_normed[j].squeeze()[i], label="Original")
    #             ax.plot(features_swapped[j].squeeze()[i], label="Target")
    #             ax.plot(features_rec[j].squeeze()[i], label="rec")
    #             # ax.set_ylim(-1,1)

    #             # Compute without silence
    #             feat_in, feat_out = features_swapped[j][i], features_rec[j][i]
    #             cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    #             correlation = (((cos(
    #                 feat_in.reshape(-1) - feat_in.mean(),
    #                 feat_out.reshape(-1) - feat_out.mean()) + 1) / 2)).item()

    #             # val = stats.pearsonr(feat_out.cpu(),feat_in.cpu())[0]

    #             ax.set_title(dataset.fader_dataset.descriptors[i] + " - " +
    #                          str(np.round(correlation, 2)))
    #             ax.legend()
    #     first = False
    #     f.savefig(run + "/descr.png")
# Save the results
torch.save(results, os.path.join(run, "results_analysis_distances.pth"))
