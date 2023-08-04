import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_stft(stft, sr, ax=None, fig=None, times=None, label=None):
    def add_colorbar(fig, ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    freqs = np.linspace(0, sr / 2, stft.shape[0])

    if times is None:
        times = np.arange(stft.shape[1])

    X, Y = np.meshgrid(times, freqs)

    im0 = ax.pcolor(X,
                    Y,
                    20 * np.log10(np.abs(stft) + 1e-6),
                    cmap="magma",
                    shading='auto')

    add_colorbar(fig, ax, im0)
    ax.set_title(label)


def plot_metrics(signal_in, signal_out, sr, n_fft=1024):
    stft_in = torch.abs(
        torch.stft(torch.tensor(signal_in),
                   n_fft=n_fft,
                   return_complex=True,
                   normalized=False,
                   window=torch.hann_window(n_fft))).numpy()

    stft_out = torch.abs(
        torch.stft(torch.tensor(signal_out),
                   n_fft=n_fft,
                   return_complex=True,
                   normalized=False,
                   window=torch.hann_window(n_fft))).numpy()

    fig = plt.figure(figsize=(20, 6))
    spec = fig.add_gridspec(2, 2)

    ax_wave_in = fig.add_subplot(spec[0, 0])
    ax_wave_out = fig.add_subplot(spec[1, 0])
    ax_stft_in = fig.add_subplot(spec[0, 1])
    ax_stft_out = fig.add_subplot(spec[1, 1])

    ax_wave_in.plot(signal_in[:1024], alpha=0.8, color='k', linewidth=0.6)
    ax_wave_out.plot(signal_out[:1024], linewidth=0.6)

    ax_wave_in.set_xlabel('Sample')
    ax_wave_in.set_ylabel('Waveform IN')
    ax_wave_out.set_xlabel('Sample')
    ax_wave_out.set_ylabel('Waveform OUT')

    fig.align_ylabels([ax_wave_in, ax_wave_out])
    # fig.tight_layout()

    plot_stft(stft_in, sr, ax_stft_in, fig, None, 'STFT IN')
    plot_stft(stft_out, sr, ax_stft_out, fig, None, 'STFT OUT')

    ax_stft_in.get_xaxis().set_visible(False)
    ax_stft_out.get_xaxis().set_visible(False)

    return fig
