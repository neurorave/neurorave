from sklearn.decomposition import PCA

import torchaudio.transforms as T
import torch
# import IPython.display as Idp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cdpam
import torch.nn.functional as F
from raving_fader.datasets.attr_dataset import get_dataset_attr
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from raving_fader.models import CRAVE
from raving_fader.helpers.core import load_config
from raving_fader.config import *
from raving_fader.pipelines import pipelines
from audio_descriptors.features import compute_all


def load_model(config_file, ckpt, datapath=None, batch_size=None):
    config = load_config(config_file)
    data_config = DataConfig(**config["data"])
    rave_config = RaveConfig(**config["rave"])
    fader_config = FaderConfig(rave=rave_config, **config["fader"])
    train_config = TrainConfig(**config["train"])

    config = BaseConfig(data=data_config, rave=rave_config, fader=fader_config, train=train_config)

    if datapath:
        config.data.preprocessed = datapath
    if batch_size:
        config.train.batch = batch_size
    pipeline = pipelines['faderave'](config)
    model = pipeline.model
    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=False)

    return config, pipeline, model, checkpoint


def forward_eval(model, audio, features):
    with torch.no_grad():
        x = audio.to(model.device).unsqueeze(1)
        x = model.pqmf(x)
        z, kl = model.reparametrize(*model.encoder(x))
        z_c = torch.cat((z, features.to(model.device)), dim=1)
        y = model.decoder(z_c, add_noise=model.warmed_up)
        y = model.pqmf.inverse(y)
        y = y.squeeze(1)
    return y


# def normalize(feat,min_max):


def spectrogram(nfft=1024, win_length=1024, hop_length=256, **kwargs):
    transform = T.Spectrogram(
        n_fft=nfft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )
    return transform


def plot_stft(stft,
              sr=16000,
              ax=None,
              fig=None,
              times=None,
              freqs=None,
              label=None):
    def add_colorbar(fig, ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if freqs is None:
        freqs = np.linspace(0, sr / 2, stft.shape[0])

    if times is None:
        times = np.arange(stft.shape[1])

    # print(stft.shape)
    X, Y = np.meshgrid(times, freqs)
    im0 = ax.pcolor(X, Y, stft, cmap="magma", shading='auto')

    add_colorbar(fig, ax, im0)
    labelRow = "Time (s)"
    labelCol = "Frequency (Hz)"
    ax.set_title(label)
    ax.set_xlabel(labelRow)
    ax.set_ylabel(labelCol)

    return fig


def test_proc(pipeline, model, audio, audio_r, features_t, imax=10000, sr=16000):
    print('Real Signal')
    Idp.display(Idp.Audio(audio, rate=sr))

    print('Rec Signal')
    Idp.display(Idp.Audio(audio_r, rate=sr))

    spec = spectrogram()(audio)
    f, ax = plt.subplots(1, 2, figsize=(16, 5))
    print(torch.log1p(spec).numpy().shape)
    plot_stft(torch.log1p(spec).numpy(), fig=f, ax=ax[0])
    ax[1].plot(audio.numpy()[:imax])

    spec_r = spectrogram()(audio_r)
    f, ax = plt.subplots(1, 2, figsize=(16, 5))
    print(torch.log1p(spec_r).numpy().shape)
    plot_stft(torch.log1p(spec_r).numpy(), fig=f, ax=ax[0])
    ax[1].plot(audio_r.numpy()[:imax])

    f, ax = plt.subplots(1, 3, figsize=(16, 5))
    features = compute_all(audio.numpy(), sr=model.sr,
                           descriptors=model.descriptors,
                           mean=False,
                           resample=pipeline.latent_length)

    features = {descr: features[descr] for descr in model.descriptors}

    features_out = compute_all(audio_r.numpy(), sr=model.sr,
                               descriptors=model.descriptors,
                               mean=False,
                               resample=pipeline.latent_length)

    features_out = {descr: features_out[descr] for descr in model.descriptors}

    print(list(features.keys()))

    features_t = dict(zip(list(features.keys()), features_t))
    print(list(features_t.keys()))
    for i, descr in enumerate(list(features.keys())):
        # features_t[descr] = pipeline.dataset.unnormalize(features_t[descr],pipeline.min_max_features[descr])
        features[descr] = pipeline.dataset.normalize(features[descr], pipeline.min_max_features[descr])
        features_out[descr] = pipeline.dataset.normalize(features_out[descr], pipeline.min_max_features[descr])
        ax[i].plot(features[descr], label='Original Features')
        ax[i].plot(features_t[descr], label='Target Features')
        ax[i].plot(features_out[descr], label='Output Features')

        ax[i].set_title(descr)
        ax[i].legend()

    plt.show()
    # print(projection_loss(spec, spec_r))


def latent_space_pca_analysis(model, test_loader, latent_dim=128):
    z_list = []
    s_list = []
    for s, _ in test_loader:
        # s_list.append(s)
        s = s.to(model.device)
        s = torch.reshape(s, (s.shape[0], 1, -1))

        # 1. multi band decomposition pqmf
        s = model.pqmf(s)

        # 2. Encode data
        mean, var = model.encoder(s)

        # z, _ = model.reparametrize(mean, var)
        z = mean
        z_list.append(z.detach().cpu())

    z_valid = torch.cat(z_list, 0)
    # print(f"nb samples : {z_valid.shape[0]}")
    z_valid = z_valid.reshape(-1, z_valid.shape[1])
    latent_mean = z_valid.mean(0)
    z_center = z_valid - latent_mean

    pca = PCA(latent_dim).fit(z_center.detach().cpu().numpy())
    components = pca.components_
    components = torch.from_numpy(components).to(z_center)

    var = pca.explained_variance_ / np.sum(pca.explained_variance_)
    var = np.cumsum(var)

    var_percent = [.8, .9, .95, .99]
    return ([np.argmax(var > p) for p in var_percent])
    # for p in var_percent:
    #     print(f"{p}%_manifold", np.argmax(var > p))


def get_mel_loss(model, x, y, nfft=2048, nmels=512, hop_length=512, win_length=2048):
    transform = T.MelSpectrogram(
        sample_rate=model.sr,
        n_fft=nfft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        # norm='slaney',
        onesided=True,
        n_mels=nmels,
    ).to(x.device)
    X_mel = transform(x)
    Y_mel = transform(y)
    loss_melspectro = F.l1_loss(X_mel, Y_mel)
    return loss_melspectro


def get_jnd_loss(model, x, y, loss_fn):
    with torch.no_grad():
        resample_rate = 22050
        resampler = T.Resample(model.sr, resample_rate, dtype=y.dtype).to(model.device)
        jnd_loss = torch.mean(loss_fn.forward(resampler(x) * 32768, resampler(y) * 32768)).item()
    return (jnd_loss)


def get_corr_attr(model, valset, loader, indices_descr, device, min_max_ref):
    feat_l1loss = dict.fromkeys(model.descriptors, 0)
    feat_corr = dict.fromkeys(model.descriptors, 0)

    for data, features in tqdm(loader):
        x = data.to(device)
        features = features.to(device)

        # features_norm = features.clone()

        features_norm = features[:, indices_descr, :].clone()

        for i, descr in enumerate(model.descriptors):
            features_norm[:, i] = valset.normalize(features_norm[:, i],
                                                   min_max_ref[descr])

            # features_norm = features_norm[:,indices_descr,:]

        if len(indices_descr) > 1:
            features_norm = features_norm[:, indices_descr, :]

        indices = np.random.choice(range(len(features_norm)), len(features_norm))
        features_swapped = torch.clone(features_norm)[indices]

        y = forward_eval(model, x, features_swapped)

        features_out = torch.zeros_like(features_swapped)

        # loud_arr =  torch.zeros(features_swapped.shape[0],features_swapped.shape[-1])

        for i, signal in enumerate(y):
            feat = compute_all(signal.cpu().numpy(), sr=model.sr,
                               descriptors=model.descriptors,
                               mean=False,
                               resample=valset.latent_length)

            for descr in model.descriptors:
                feat[descr] = valset.normalize(feat[descr], min_max_ref[descr])

            feat = {descr: feat[descr] for descr in model.descriptors}
            feature_arr = np.array(list(feat.values())).astype(np.float32)
            features_out[i] = torch.tensor(feature_arr)

            #         loud = get_loudness(signal.cpu().numpy(),model.sr,256,1024)
        #         loud = librosa.core.resample(loud,len(loud),features_swapped.shape[-1])
        #         loud_arr[i] =  torch.tensor(loud)

        for i, descr in enumerate(model.descriptors):
            feat_out = features_out[:, i]
            feat_in = features_swapped[:, i]
            feat_l1loss[descr] += F.l1_loss(feat_in, feat_out).item() / len(loader)
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            feat_corr[descr] += (
                        cos(feat_in.reshape(-1) - feat_in.mean(), feat_out.reshape(-1) - feat_out.mean()) / len(
                    loader)).item()

    return (feat_corr, feat_l1loss)

    # torch.corrcoef(torch.stack((feat_in.reshape(-1),feat_out.reshape(-1))))


def get_evalset(eval_path, config_file, batch, eval_file=None):
    config = load_config(config_file)
    data_config = DataConfig(**config["data"])
    rave_config = RaveConfig(**config["rave"])
    fader_config = FaderConfig(rave=rave_config, **config["fader"])
    train_config = TrainConfig(**config["train"])

    config = BaseConfig(data=data_config, rave=rave_config, fader=fader_config, train=train_config)
    latent_length = config.data.n_signal // (config.rave.data_size * np.prod(config.rave.ratios))

    print(fader_config)
    try:
        if eval_file == None:
            allfeatures = torch.load(eval_path + "/eval_features.pth")
        else:
            allfeatures = torch.load(os.path.join(eval_path, eval_file))
        print('features imported')
    except:
        allfeatures = None

    dataset_eval = get_dataset_attr(
        preprocessed=eval_path,
        wav=eval_path,
        sr=config.data.sr,
        descriptors=config.data.descriptors,
        n_signal=config.data.n_signal,
        latent_length=latent_length,
        r_samples=None,
        nb_bins=config.fader.num_classes,
        allfeatures=allfeatures)

    allfeatures = torch.save(dataset_eval.allfeatures, eval_path + "/eval_features.pth")

    eval_loader = DataLoader(dataset_eval,
                             batch,
                             True,
                             drop_last=True,
                             num_workers=8)
    return (dataset_eval, eval_loader)

# def compute_features(signal):


#     # loud_arr =  torch.zeros(features_swapped.shape[0],features_swapped.shape[-1])

#     features_out = torch.zeros_like(features_swapped)
#     for i,signal in enumerate(y):
#         feat = compute_all(signal.cpu().numpy(),sr=model.sr,
#                                                 descriptors=model.descriptors,
#                                                 mean=False,
#                                                 resample=valset.latent_length)
#         # for descr in model.descriptors:
#         #     feat[descr] = valset.normalize(feat[descr],min_max_ref[descr])

#         feat = {descr: feat[descr] for descr in model.descriptors}
#         feature_arr = np.array(list(feat.values())).astype(np.float32)
#             features_out[i] = torch.tensor(feature_arr)
