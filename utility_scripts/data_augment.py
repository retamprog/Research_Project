import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NOISE_FACTOR,TIME_MASK,FREQ_MASK

def add_random_noise(audio,noise_factor:float=NOISE_FACTOR):
    # adding random gaussian noise to simualate real world situations
    noise = np.random.randn(len(audio))*noise_factor # here the random noise data is scaled by the noise factor
    return (audio+noise).astype(np.float32)

def time_shift_with_noise(audio:np.ndarray,shift_max=0.1,sr:int=16000,noise_level:float=0.001):
    '''
    This function randomly shifts audio along the time line based on the max shift = 0.1 
    while replacing the empty regions with low -level gaussian background noise
    '''
    max_shift = int(sr*shift_max)

    #random shift amount
    shift = np.random.randint(-max_shift,max_shift)

     # Create empty output array
    shifted = np.zeros_like(audio)

    if shift > 0:
        # Shift right
        shifted[shift:] = audio[:-shift]
        # Fill beginning with background noise
        shifted[:shift] = noise_level * np.random.randn(shift)

    elif shift < 0:
        # Shift left
        shifted[:shift] = audio[-shift:]
        # Fill ending with background noise
        shifted[shift:] = noise_level * np.random.randn(-shift)

    else:
        shifted = audio

    return shifted.astype(np.float32)

def speed_perturb(audio,rate_change:tuple=(0.9,1.1)):
    # Slightly change the playback speed (pitch independent stretch).
    rate =np.random.uniform(*rate_change)
    n_new = int(len(audio)/rate)
    idx = np.linspace(0,len(audio)-1,n_new)
    return np.interp(idx,np.arange(len(audio)),audio).astype(np.float32)

def apply_time_mask(feature: np.ndarray, T: int = TIME_MASK) -> np.ndarray:
    """
    SpecAugment: mask a random consecutive chunk of time frames.
    Input shape: (freq, time) or (1, freq, time)
    """
    feat = feature.copy()
    time_len = feat.shape[-1]
    t = np.random.randint(0, T)
    t0 = np.random.randint(0, max(time_len - t, 1))
    feat[..., t0:t0+t] = 0.0
    return feat


def apply_freq_mask(feature: np.ndarray, F: int = FREQ_MASK) -> np.ndarray:
    """
    SpecAugment: mask a random chunk of frequency bins.
    Input shape: (freq, time) or (1, freq, time)
    """
    feat = feature.copy()
    freq_len = feat.shape[-2]
    f = np.random.randint(0, F)
    f0 = np.random.randint(0, max(freq_len - f, 1))
    feat[..., f0:f0+f, :] = 0.0
    return feat


def augment_audio(audio: np.ndarray, sr: int = 16000, p: float = 0.5) -> np.ndarray:
    """
    Apply random audio-level augmentations.
    Each augmentation is applied independently with probability p.
    """
    if np.random.rand() < p:
        audio = add_random_noise(audio)
    if np.random.rand() < p * 0.5:
        audio = time_shift_with_noise(audio, sr=sr)
    if np.random.rand() < p * 0.3:
        audio = speed_perturb(audio)
    return audio


def augment_feature(feature: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Apply SpecAugment-style feature-level augmentations.
    Applied during training after feature extraction.
    """
    if np.random.rand() < p:
        feature = apply_time_mask(feature)
    if np.random.rand() < p:
        feature = apply_freq_mask(feature)
    return feature

