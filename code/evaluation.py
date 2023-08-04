import os
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm
# import matplotlib.pyplot as plt
import numpy as np
from raving_fader.helpers.evaluation_functions import load_model, get_features, normalize, swap_features
from effortless_config import Config
import cdpam
from raving_fader.helpers.eval import get_jnd_loss, get_mel_loss

torch.set_grad_enabled(False)


### CONFIG ###
class args(Config):
    MODELNAME = None
    STEP = None


first = True

args.parse_args()

# Params
models_dir = "/data/nils/raving-fader/models/"  #All models directory
model_name = "FRAVE_NSS"  #Model name
step = 1000000
nb_swap = 5  # Number of attributes to swap

if args.MODELNAME is not None:
    model_name = args.MODELNAME
if args.STEP is not None:
    step = args.STEP

## Import the model
path = os.path.join(models_dir, model_name)
ckpt = os.path.join(path, model_name + "__vae_" + str(step) + ".ckpt")
# ckpt = os.path.join(path, model_name + "__vae" + ".ckpt")
config_file = os.path.join(path, "train_config.yaml")

config, pipeline, model, checkpoint = load_model(config_file,
                                                 ckpt,
                                                 datapath=None,
                                                 batch_size=8)

print("Model loaded : ", model_name, " - Chekpoint at step :",
      checkpoint["step"])

## Get the validation set dataloader
validloader = pipeline.val_set

# Import the features from the full dataset
dataset_dir = config.data.preprocessed
all_features = torch.load(os.path.join(dataset_dir, "features.pth"))

##########  COMPUTATION LOOP ############

#Create the results dict
results = {
    "distance": 0,
    "mel": 0,
    "jnd": 0,
    "correlation": {descr: 0
                    for descr in model.descriptors},
    "l1_loss": {descr: 0
                for descr in model.descriptors}
}

# Resampler for timbral models descriptors
resampler = T.Resample(model.sr, 44100).to(model.device)  #
loss_fn_jnd = cdpam.CDPAM(dev=model.device)

# Iterate over the validation set
for k, (x, features) in enumerate(tqdm(validloader)):
    # Get x and features from dataloader
    x = x.unsqueeze(1).to(model.device)

    #Normalize the features
    features_normed = normalize(features, all_features)
    features_normed = features_normed

    #Create the swapped features
    features_swapped = swap_features(all_features, features_normed, nb_swap)

    #Reconstruct the signals
    z, kl = model.reparametrize(*model.encoder(model.pqmf(x)))

    z_c = torch.cat((z, features_normed.to(model.device)), dim=1)
    y = model.decoder(z_c, add_noise=False)
    y = model.pqmf.inverse(y)

    #Compute the losses
    distance = model.distance(
        x, y).item()  # Multiscale STFT distance (training loss)

    distance_mel = get_mel_loss(model,
                                x,
                                y,
                                nfft=2048,
                                nmels=512,
                                hop_length=512,
                                win_length=2048).item()
    jnd = get_jnd_loss(model, x.squeeze(), y.squeeze(), loss_fn_jnd)

    results["distance"] += distance / len(validloader)
    results["jnd"] += jnd / len(validloader)
    results["mel"] += distance_mel / len(validloader)

    z_c_mod = torch.cat((z, features_swapped.to(model.device)), dim=1)
    y_mod = model.decoder(z_c_mod, add_noise=False)
    y_mod = model.pqmf.inverse(y_mod)

    #Resample using torchaudio for faster feature computation

    #Recompute the features
    features_rec = torch.zeros_like(features_normed)
    # if the model sample rate is below 44100, resample the signal with torchaudio for faster computation of timnbra descriptors
    if model.sr < 44100:
        y_mod_resamp = resampler(y_mod)

        for i, (signal, signal_resamp) in enumerate(
                zip(y_mod.squeeze().detach().cpu().numpy(),
                    y_mod_resamp.squeeze().detach().cpu().numpy())):
            features_rec[i, :3] = get_features(
                signal, model.descriptors[:3],
                sr=model.sr)  #Using the raw signal for librosa descriptors
            features_rec[i, 3:] = get_features(
                signal_resamp, model.descriptors[3:], sr=44100
            )  #Using the torchaudio-resampled signal for timbra models (faster)

    else:
        for i, signal in enumerate(y_mod.squeeze().detach().cpu().numpy()):
            features_rec[i, :] = get_features(signal,
                                              model.descriptors,
                                              sr=model.sr)

    features_rec = normalize(features_rec, all_features)

    for i, descr in enumerate(model.descriptors):
        feat_out = features_rec[:, i]
        feat_in = features_swapped[:, i]

        l1_loss_descr = F.l1_loss(feat_in, feat_out).item()
        results["l1_loss"][descr] += l1_loss_descr / len(validloader)

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        correlation = (((cos(
            feat_in.reshape(-1) - feat_in.mean(),
            feat_out.reshape(-1) - feat_out.mean()) + 1) / 2)).item()
        results["correlation"][descr] += correlation / len(validloader)

    # if first == True:
    #     f, axs = plt.subplots(8, 5, figsize=(20, 80))
    #     features_swapped = features_swapped.cpu()
    #     features_normed = features_normed.cpu()
    #     features_rec = features_rec.cpu()
    #     for j in range(8):
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

    #             ax.set_title(model.descriptors[i] + " - " +
    #                          str(np.round(correlation, 2)))
    #             ax.legend()
    #     first = False
    #     print(results)
    #     f.savefig(path + "/descr.png")

# Save the results
torch.save(
    results,
    os.path.join(models_dir, model_name) +
    "/results_analysis_distances_1M.pth")
