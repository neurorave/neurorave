"""

 ~ Generative audio metrics ~
 features.py : Compute audio features
 
 This file contains simple functions for computing audio features from a 
 given input sample. It relies on the libraries
 - Librosa
 - Timbral (audio commons) models (modified to provide time series)
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
 
 All authors contributed equally to the project.
 
"""
import librosa
import numpy as np
from generative_metrics.audio.timbral_models import (
    timbral_booming,
    timbral_brightness,
    timbral_depth,
    timbral_hardness,
    timbral_reverb,
    timbral_roughness,
    timbral_sharpness,
    timbral_warmth)


def compute_timbral(y: np.ndarray,
                    sr: int,
                    mean: bool = False) -> dict:
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
        # "hardness": timbral_hardness,
        "roughness": timbral_roughness,
        "sharpness": timbral_sharpness,
        "warmth": timbral_warmth
    }
    # Results dict
    features = {}
    if sr < 44100:
        # upsample file to avoid errors 
        y = librosa.core.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100
    for name, func in features_dict.items():
        features[name] = func(y, fs=sr, mean=mean)
    return features


def compute_librosa(y: np.ndarray,
                    sr: int,
                    mean: bool = False) -> dict:
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
    features["rms"] = librosa.feature.rms(y=y)
    features["zcr"] = librosa.feature.zero_crossing_rate(y)
    features["f0"] = librosa.yin(y, fmin=50, fmax=5000, sr=sr)[np.newaxis, :]
    # Spectral features
    S, phase = librosa.magphase(librosa.stft(y=y))
    # Compute all descriptors
    for name, func in features_dict.items():
        features[name] = func(S=S)
    return features


def compute_all(x: np.ndarray,
                sr: int,
                mean: bool = False,
                resample: bool = True) -> dict:
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
    librairies = {
        "librosa": compute_librosa,
        "timbral": compute_timbral
    }
    # Final features
    final_features = {}
    for n, func in librairies.items():
        # Process all functions
        cur_dict = func(x, sr, mean)
        # Merge dictionnaries
        final_features.update(cur_dict)
    # Resample all series to max length
    if (resample):
        m_len = max([x.shape[0] for x in final_features.values()])
        for n, v in final_features.items():
            final_features[n] = librosa.core.resample(v, orig_sr=v.shape[0], target_sr=m_len)
    return final_features


if __name__ == '__main__':
    sr = 480000
    vals = np.load('generative_metrics/audio/failures.npy')
    for x in vals:
        print(x)
        # Load a test sound
        # x, sr = librosa.core.load('toy/test_sound.wav')
        # Compute all features
        # print(np.mean(x + (np.random.randn(x.shape[0]) * 1e-9)))
        features = compute_all(np.array(x) + 1e-9, sr)  # + (np.random.randn(x.shape[0]) * 1e-9), sr)
    # print(features)
