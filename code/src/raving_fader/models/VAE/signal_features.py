import torch
import librosa
from utils import griffin_lim, mel_to_stft
import torchaudio
import numpy as np
# -----------------------------------------------------------
#
# Signal features computation
#
# -----------------------------------------------------------

# Set of computable signal features
features = {
    'duration': (librosa.get_duration, 'float'),
    # Harmonics
    'salience': (librosa.salience, 'ndarray'),
    'pyin': (librosa.yin, 'ndarray'),
    # Spectral Features
    'rms': (librosa.feature.rms, 'ndarray'),
    'spectral_centroid': (librosa.feature.spectral_centroid, 'ndarray'),
    'spectral_bandwidth': (librosa.feature.spectral_bandwidth, 'ndarray'),
    'spectral_contrast': (librosa.feature.spectral_contrast, 'ndarray'),
    'spectral_flatness': (librosa.feature.spectral_flatness, 'ndarray'),
    'spectral_rolloff': (librosa.feature.spectral_rolloff, 'ndarray'),
    'zero_crossing_rate': (librosa.feature.zero_crossing_rate, 'ndarray'),
}

features_simple = features
"""
features_simple = {
    'duration': (librosa.get_duration, 'float'),
    'salience': (librosa.salience, 'ndarray'),
    'pyin': (librosa.pyin, 'ndarray'),
    'localmax': (librosa.util.localmax, 'ndarray'),
    'localmin': (librosa.util.localmin, 'ndarray'),
    'peak_pick': (librosa.util.peak_pick, 'ndarray'),
}
"""


def dataset_features(x_cur, sr, feature_set=None):
    if feature_set is None:
        feature_set = features
    feature_vals = {}
    # Perform the extraction
    for key, val in feature_set.items():
        if val[0] is None:
            continue
        if key in ['spectral_flatness', 'zero_crossing_rate']:
            value = (val[0](x_cur))
        elif key == 'salience':
            S = np.abs(librosa.stft(x_cur))
            freqs = librosa.core.fft_frequencies(sr)
            harms = [1, 2, 3, 4]
            value = (val[0](S, freqs, harms))
            value[np.isnan(value)] = 0
        elif key == 'pyin':
            value = (val[0](x_cur, fmin=50, fmax=librosa.note_to_hz('C7')))
        else:
            value = (val[0](x_cur, sr))
        if type(value) is not float:
            value = np.mean(value)
        feature_vals[key] = value
    return feature_vals


# Function to compute features of one sample
def signal_features(x_cur, config, feature_set=None):
    # First create a wav version
    if feature_set is None:
        feature_set = features
    if config["data"]["representation"] in ['spectrogram', 'melspectrogram']:
        x_cur = torch.exp(x_cur) - 1
        if config["data"]["representation"] == 'melspectrogram':
            x_cur = mel_to_stft(x_cur, config)
        wav = griffin_lim(x_cur, config)
        torchaudio.save('/tmp/sample.wav', wav.cpu(),
                        config["preprocess"]["sample_rate"])
    elif config["data"]["representation"] == 'waveform':
        torchaudio.save('/tmp/sample.wav', x_cur.cpu(),
                        config["preprocess"]["sample_rate"])
    x_cur, sr = librosa.load('/tmp/sample.wav')
    feature_vals = {}
    # Perform all desired extraction
    for key, val in feature_set.items():
        if val[0] is None:
            continue
        if key in ['spectral_flatness', 'zero_crossing_rate']:
            value = (val[0](x_cur))
        elif key == 'salience':
            S = np.abs(librosa.stft(x_cur))
            freqs = librosa.core.fft_frequencies(sr)
            harms = [1, 2, 3, 4]
            value = (val[0](S, freqs, harms))
            value[np.isnan(value)] = 0
        elif key == 'pyin':
            value = (val[0](x_cur, fmin=50, fmax=librosa.note_to_hz('C7')))
        else:
            value = (val[0](x_cur, sr))
        if type(value) is not float:
            value = np.mean(value)
        feature_vals[key] = value
    return feature_vals


# Function to compute all features from a given loader
def compute_signal_features(loader, config):
    final_features = {}
    for f in features:
        final_features[f] = []
    for x in loader:
        # Send to device
        x = x.to(config["train"]["device"], non_blocking=True)
        for x_cur in x:
            # print(x_cur.mean())
            # Compute signal features on input
            feats = signal_features(x_cur, config)
            for f in features:
                final_features[f].append(feats[f])
    return final_features
