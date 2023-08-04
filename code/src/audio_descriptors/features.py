"""

 features.py : Compute audio features
 
 This file contains simple functions for computing audio features from a 
 given input sample. It relies on the libraries
 - Librosa
 - Timbral (audio commons) models (modified to provide time series)
 
 Author               :  Philippe Esling, Giovanni Bindi
                        <esling@ircam.fr, bindi@ircam.fr>
 
 All authors contributed equally to the project.
 
"""
import librosa
import numpy as np
from audio_descriptors.timbral_models import (
    timbral_booming, timbral_brightness, timbral_depth, timbral_hardness,
    timbral_reverb, timbral_roughness, timbral_sharpness, timbral_warmth)
import torchaudio.transforms as T
import torch


def compute_timbral(y: np.ndarray,
                    sr: int,
                    descriptors: list = [None],
                    mean: bool = False,
                    resampler=None) -> dict:
    """
    Compute all descriptors inside the timbral models (audio common) library

    Parameters
    ----------
    x : np.ndarray
        Input audio signal (samples)
    sr : int
        Input sample rate
    mean : bool, optional
        [TODO] : Compute the mean of descriptors

    Returns
    -------
    dict
        Dictionnary containing all features.

    """
    # Features to compute
    features_dict = {
        "booming": timbral_booming,
        "brightness": timbral_brightness,
        "depth": timbral_depth,
        "hardness": timbral_hardness,
        "roughness": timbral_roughness,
        "sharpness": timbral_sharpness,
        "warmth": timbral_warmth
    }
    # Results dict
    features = {}
    if max([(feature in descriptors)
            for feature in list(features_dict.keys())]):
        if sr < 44100:
            if resampler is None:
                y = librosa.core.resample(y, orig_sr=sr, target_sr=44100)
                sr = 44100
            else:
                # upsample file to avoid errors
                y = resampler(torch.tensor(y)).numpy()
                sr = 44100
        for name, func in features_dict.items():
            if name in descriptors:
                features[name] = func(y, fs=sr, mean=mean)
    return features


def compute_librosa(y: np.ndarray,
                    sr: int,
                    descriptors: list = [None],
                    mean: bool = False,
                    resampler=None) -> dict:
    """
    Compute all descriptors inside the Librosa library

    Parameters
    ----------
    x : np.ndarray
        Input audio signal (samples)
    sr : int
        Input sample rate
    mean : bool, optional
        [TODO] : Compute the mean of descriptors

    Returns
    -------
    dict
        Dictionnary containing all features.

    """
    # Features to compute
    features_dict = {
        "rolloff": librosa.feature.spectral_rolloff,
        "flatness": librosa.feature.spectral_flatness,
        "bandwidth": librosa.feature.spectral_bandwidth,
        "centroid": librosa.feature.spectral_centroid
    }
    # Results dict
    features = {}
    # Temporal features
    if "rms" in descriptors:
        features["rms"] = librosa.feature.rms(y=y)
    if "zcr" in descriptors:
        features["zcr"] = librosa.feature.zero_crossing_rate(y)
    if "f0" in descriptors:
        features["f0"] = librosa.yin(y, fmin=50, fmax=5000,
                                     sr=sr)[np.newaxis, :]
    # Spectral features
    S, phase = librosa.magphase(librosa.stft(y=y))
    # Compute all descriptors

    for name, func in features_dict.items():
        if name in descriptors:
            features[name] = func(S=S)
    return features


def compute_all(x: np.ndarray,
                sr: int,
                descriptors: list = [None],
                mean: bool = False,
                resample=None,
                resampler=None) -> dict:
    """
    Compute all descriptors inside a given dictionnary of function. This
    high-level launch computations and merge dictionnaries.
    Finally allows to resample all descriptor series to a common length.

    Parameters
    ----------
    x : np.ndarray
        Input audio signal (samples)
    sr : int
        Input sample rate
    mean : bool, optional
        [TODO] : Compute the mean of descriptors
    resample : bool, optional
        Resample all series to the maximum length found. The default is True.

    Returns
    -------
    dict
        Dictionnary containing all features.

    """
    # List of feature sub-libraries to use
    librairies = {"librosa": compute_librosa, "timbral": compute_timbral}
    # Final features
    final_features = {}
    for n, func in librairies.items():
        # Process all functions
        cur_dict = func(x, sr, descriptors, mean, resampler)
        # Merge dictionnaries
        final_features.update(cur_dict)
    # Resample all series to max length
    if resample:
        # m_len = max([x.shape[0] for x in final_features.values()])
        for n, v in final_features.items():
            if len(v.shape) > 1:
                v = v[0]
            final_features[n] = librosa.core.resample(v,
                                                      orig_sr=v.shape[0],
                                                      target_sr=resample)
            final_features[n].shape
    return final_features


# def compute_all(x: np.ndarray,
#                 sr: int,
#                 descriptors: list = [None],
#                 mean: bool = False,
#                 resample=None) -> dict:
#     """
#     Compute all descriptors inside a given dictionnary of function. This
#     high-level launch computations and merge dictionnaries.
#     Finally allows to resample all descriptor series to a common length.

#     Parameters
#     ----------
#     x : np.ndarray
#         Input audio signal (samples)
#     sr : int
#         Input sample rate
#     mean : bool, optional
#         [TODO] : Compute the mean of descriptors
#     resample : bool, optional
#         Resample all series to the maximum length found. The default is True.

#     Returns
#     -------
#     dict
#         Dictionnary containing all features.

#     """
#     # List of feature sub-libraries to use
#     librairies = {"librosa": compute_librosa, "timbral": compute_timbral}
#     # Final features
#     final_features = {}
#     for n, func in librairies.items():
#         # Process all functions
#         cur_dict = func(x, sr, descriptors, mean)
#         # Merge dictionnaries
#         final_features.update(cur_dict)
#     # Resample all series to max length
#     if resample:
#         # m_len = max([x.shape[0] for x in final_features.values()])
#         for n, v in final_features.items():
#             if len(v.shape) > 1:
#                 v = v[0]
#             final_features[n] = librosa.core.resample(v,
#                                                       orig_sr=v.shape[0],
#                                                       target_sr=resample)
#             final_features[n].shape
#     return final_features

if __name__ == '__main__':
    # Load a test sound
    x, sr = librosa.core.load('./toy/test_sound.wav')
    # Compute all features
    features = compute_all(x, sr)
    # print(features)
