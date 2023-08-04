import torch
import matplotlib.pyplot as plt
import os
import librosa
import torchaudio.transforms as T
from logging import getLogger

global tr_mel
tr_mel = None

logger = getLogger()


##
# -----------------------------------------------------------
#
# Features Extraction
#
# -----------------------------------------------------------
def spectrogram(x, config):
    computed_spectrogram = T.Spectrogram(
        n_fft=config["preprocess"]["n_fft"],
        win_length=config["preprocess"]["win_length"],
        hop_length=config["preprocess"]["hop_length"],
        center=True,
        pad_mode="reflect",
        power=2.0,
    )
    computed_spectro = computed_spectrogram(x)
    return computed_spectro


def griffin_lim(x, config):
    computed_griffin_lim = T.GriffinLim(
        n_fft=config["preprocess"]["n_fft"],
        win_length=config["preprocess"]["win_length"],
        hop_length=config["preprocess"]["hop_length"],
        n_iter=32).to(x.device)
    computed_griffin_lim = computed_griffin_lim(x)
    return computed_griffin_lim


def mel_to_stft(x_cur, config):
    global tr_mel
    if tr_mel is None:
        # print(x_cur.shape)
        # print(config["preprocess"]["n_mels"])
        tr_mel = T.InverseMelScale(
            n_stft=int(config["preprocess"]["n_fft"] / 2) + 1,
            n_mels=config["preprocess"]["n_mels"],
            max_iter=10).to(config["train"]["device"])
    return tr_mel(x_cur)


def init_mel_transform(config):
    transform = T.MelSpectrogram(
        sample_rate=config["preprocess"]["sample_rate"],
        n_fft=config["preprocess"]["n_fft"],
        win_length=config["preprocess"]["win_length"],
        hop_length=config["preprocess"]["hop_length"],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=config["preprocess"]["n_mels"],
    ).cpu()
    return transform


def mel_spectrogram(x, transform, config):
    return transform(x)


##
# -----------------------------------------------------------
#
# Audio stats and metadata
#
# -----------------------------------------------------------
def plot_specgram(spectro_dir,
                  waveform,
                  sample_rate,
                  title="Spectrogram",
                  xlim=None,
                  save=True):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    # plt.show(block=True)
    if save:
        spec_save = title
        if not os.path.exists(spectro_dir):
            os.mkdir(os.path.join(spectro_dir, spec_save))
        plt.savefig(os.path.join(spectro_dir, spec_save))


def plot_spectrogram(spec,
                     spectro_dir='spectro_dir',
                     title=None,
                     ylabel='freq_bin',
                     aspect='auto',
                     xmax=None,
                     show=False,
                     save=True):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    if show:
        plt.show()
    if save:
        spec_save = title
        if not os.path.exists(spectro_dir):
            os.mkdir(os.path.join(spectro_dir, spec_save))
        plt.savefig(os.path.join(spectro_dir, spec_save))
    return im


def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def print_metadata(metadata, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    print(" - sample_rate:", metadata.sample_rate)
    print(" - num_channels:", metadata.num_channels)
    print(" - num_frames:", metadata.num_frames)
    print(" - bits_per_sample:", metadata.bits_per_sample)
    print(" - encoding:", metadata.encoding)
    print()


def clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
    """
    parameters = list(parameters)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm**norm_type
        total_norm = total_norm**(1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef >= 1:
        return
    for p in parameters:
        p.grad.data.mul_(clip_coef)


def print_accuracies(values):
    """
    Pretty plot of accuracies.
    """
    assert all(len(x) == 2 for x in values)
    for name, value in values:
        logger.info('{:<20}: {:>6}'.format(name, '%.3f%%' % (100 * value)))
    logger.info('')


def get_lambda(l, config):
    """
    Compute discriminators' lambdas.
    """
    s = config["discriminator"]["lambda_schedule"]
    if s == 0:
        return l
    else:
        return l * float(
            min(config["train"]["epochs"] * config["train"]["batch_size"],
                s)) / s


# A set of mu law encode/decode functions
def mu_law_encode(signal, quantization_channels, config):
    # Manual mu-law companding and mu-bits quantization
    mu = torch.tensor([quantization_channels - 1]).to(signal, non_blocking=True)
    # mu = mu.to(config["train"]["device"])
    magnitude = torch.log1p(mu * torch.abs(signal)) / torch.log1p(mu)
    signal = torch.sign(signal) * magnitude

    # Map signal from [-1, +1] to [0, mu-1]
    signal = (signal + 1) / 2 * mu + 0.5

    return signal.long()


def merge_tensors(tensor_1, tensor_2, way):
    if way == "expand":  # TODO
        output = [tensor_1, tensor_2.expand(16, 8)]
    elif way == "cat":
        # Resulting tensor will have dim0 unchanged, an addition is performed for dim1
        output = torch.cat((tensor_1, tensor_2), dim=1)
    else:
        output = None
    return output
