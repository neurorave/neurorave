import torch
import numpy as np

from audio_descriptors.features import compute_all
from raving_fader.helpers.eval import load_model
from raving_fader.config import *
from raving_fader.helpers.core import load_config
from raving_fader.pipelines import pipelines


def load_model(config_file, ckpt, datapath=None, batch_size=None):
    config = load_config(config_file)
    data_config = DataConfig(**config["data"])
    rave_config = RaveConfig(**config["rave"])
    fader_config = FaderConfig(rave=rave_config, **config["fader"])
    train_config = TrainConfig(**config["train"])

    config = BaseConfig(data=data_config,
                        rave=rave_config,
                        fader=fader_config,
                        train=train_config)

    if datapath:
        config.data.preprocessed = datapath
    if batch_size:
        config.train.batch = batch_size
    pipeline = pipelines['faderave'](config)
    model = pipeline.model
    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"],
                                  strict=False)

    return config, pipeline, model, checkpoint


def get_features(audio, descriptors, sr, latent_length=64):
    feat = compute_all(audio,
                       sr=sr,
                       descriptors=descriptors,
                       mean=False,
                       resample=latent_length)

    feat = {descr: feat[descr] for descr in descriptors}
    feat = np.array(list(feat.values())).astype(np.float32)
    return torch.tensor(feat)


# def plot_features(features, descriptors, features_2=None):
#     f, axs = plt.subplots(1, 5, figsize=(20, 5))
#     for i, ax in enumerate(axs):
#         ax.plot(features[i], label="Original")

#         if features_2 is not None:
#             ax.plot(features_2[i], label="Modified")
#         # ax.set_ylim(-1,1)
#         ax.set_title(descriptors[i])
#         ax.legend()


def normalize(features, all_features):
    features_normed = torch.clone(features)
    for i in range(features.shape[1]):
        features_normed[:, i] = 2 * (
            (features[:, i] - np.min(all_features[:, i])) /
            (np.max(all_features[:, i]) - np.min(all_features[:, i])) - 0.5)
    return features_normed


def create_funcs(features_swapped):
    n_points = 64
    # Lots of basic funcs
    zeros = np.zeros(n_points)
    ones = np.ones(n_points)
    ones_m = np.ones(n_points) * -1
    linear = np.linspace(-1, 1, n_points)
    linear_inv = np.linspace(1, -1, n_points)
    # Selector array
    basic_funcs = [zeros, ones, ones_m, linear, linear_inv]
    # Sinusoid generator
    sin_funcs = []
    for i in range(10):
        freq = np.random.randint(1, 30)
        phase = np.random.randn() * 2 * np.pi
        sin_rnd = np.sin((linear * freq) + phase)
        sin_funcs.append(sin_rnd)

    # Interpolator
    sel_1 = basic_funcs[np.random.randint(0, len(basic_funcs))]
    sel_2 = sin_funcs[np.random.randint(0, len(sin_funcs))]
    alphas = np.linspace(0., 1., 10)
    int_funcs = []
    for alpha in alphas:
        int_sig = alpha * sel_2 + ((1 - alpha) * sel_1)
        int_funcs.append(int_sig)

    # Interpolator

    sel_1, sel_2 = features_swapped[0], features_swapped[1]
    # plt.plot(sel_1)
    # plt.plot(sel_2)
    alphas = np.linspace(0., 1., 50)
    eval_funcs = []
    for alpha in alphas:
        int_sig = alpha * sel_2 + ((1 - alpha) * sel_1)
        eval_funcs.append(int_sig)

    full_funcs = []
    for funcs in [basic_funcs, sin_funcs, int_funcs]:
        for func in funcs:
            full_funcs.append(torch.tensor(func))
    for func in eval_funcs:
        full_funcs.append(func)

    full_funcs = torch.stack(full_funcs)
    return full_funcs


def swap_features(all_features, features_normed, nb_swap):

    features_swapped = torch.clone(features_normed)

    id_random = np.random.choice(range(len(all_features)), 2)
    features_ext = torch.tensor(all_features[id_random, :])
    features_ext = normalize(features_ext, all_features)

    id_descr = np.random.choice([i for i in range(5)], nb_swap, replace=False)
    # id_descr = [0]
    for j in id_descr:
        features_pool = create_funcs(features_ext[:, j])
        id_random = np.random.choice(range(len(features_pool)),
                                     len(features_normed))

        features_swapped[:, j] = features_pool[id_random, :].squeeze(1)
    return features_swapped
