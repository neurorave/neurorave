import numpy as np
from random import random
from scipy.signal import lfilter

from udls import simple_audio_preprocess, SimpleDataset
from udls.transforms import Compose, RandomApply, Dequantize, RandomCrop


def random_angle(min_f=20, max_f=8000, sr=24000):
    min_f = np.log(min_f)
    max_f = np.log(max_f)
    rand = np.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * np.pi * rand / sr
    return rand


def pole_to_z_filter(omega, amplitude=.9):
    z0 = amplitude * np.exp(1j * omega)
    a = [1, -2 * np.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * np.real(z0), 1]
    return b, a


def random_phase_mangle(x, min_f, max_f, amp, sr):
    angle = random_angle(min_f, max_f, sr)
    b, a = pole_to_z_filter(angle, amp)
    return lfilter(b, a, x)


def get_dataset(preprocessed, wav, sr, n_signal):
    dataset = SimpleDataset(
        preprocessed,
        wav,
        # len_signal =  2 * n_signal,
        preprocess_function=simple_audio_preprocess(sr, 2 * n_signal),
        split_set="full",
        # map_size=1e8,
        transforms=Compose([
            RandomCrop(n_signal),
            RandomApply(
                lambda x: random_phase_mangle(x, 20, 2000, .99, sr),
                p=.8,
            ),
            Dequantize(16),
            lambda x: x.astype(np.float32),
        ]),
    )
    return dataset
