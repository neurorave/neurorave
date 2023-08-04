import torch
import librosa as li
from random import random, choice, randint
import numpy as np
import matplotlib.pyplot as plt


class Transform(object):
    def __call__(self, x: torch.Tensor):
        raise NotImplementedError


class RandomApply(Transform):
    def __init__(self, transform, p=.5):
        self.transform = transform
        self.p = p

    def __call__(self, x: np.ndarray):
        if random() < self.p:
            x = self.transform(x)
        return x


class Compose(Transform):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x: np.ndarray):
        for elm in self.transform_list:
            x = elm(x)


class RandomChoice(Transform):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x: np.ndarray):
        x = choice(self.transform_list)(x)
        return x


class PitchShift(Transform):
    def __init__(self, mean, std, sr):
        self.mean = mean
        self.std = std
        self.sr = sr

    def __call__(self, x: np.ndarray):
        r = self.std * (random() - .5) + self.mean
        x = li.effects.pitch_shift(x, self.sr, r, res_type="kaiser_fast")
        return x


class Reverb(Transform):
    def __init__(self, mean, std, sr):
        self.mean = mean
        self.std = std
        self.sr = sr

    def __call__(self, x: np.ndarray):
        r = self.std * (random() - .5) + self.mean

        noise = 2 * np.random.rand(self.sr) - 1
        fade = np.linspace(1, 0, self.sr)
        exp = np.exp(-np.linspace(0, r, self.sr))

        impulse = noise * fade * exp
        impulse[0] = 1

        shape_x_ori = len(x)
        N = max(len(x), len(impulse))
        x = np.pad(x, (0, N - len(x)))
        impulse = np.pad(impulse, (0, N - len(impulse)))

        y = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(impulse))
        y = y[:shape_x_ori]

        return y


class Noise(Transform):
    def __init__(self, std):
        self.std = std

    def __call__(self, x: np.ndarray):
        return x + self.std * (2 * np.random.rand(len(x)) - 1)


class RandomCrop(Transform):
    def __init__(self, n_signal):
        self.n_signal = n_signal

    def __call__(self, x: np.ndarray):
        in_point = randint(0, len(x) - self.n_signal)
        x = x[in_point:in_point + self.n_signal]
        return x


class Reverse(Transform):
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray):
        return -x


class Dequantize(Transform):
    def __init__(self, bit_depth):
        self.bit_depth = bit_depth

    def __call__(self, x: np.ndarray):
        x += np.random.rand(len(x)) / 2**self.bit_depth
        return x
