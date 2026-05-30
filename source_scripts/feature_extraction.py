# Python script for feature extraction functions  and particular custom functions for feeding it to our CNN models

# Two Features
# 1> MFCC (Mel frequency cepstral coefficients)
# 2>Log Mel
import os
import librosa
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from config import (
    SAMPLE_RATE, MFCC_N_FFT,MFCC_N_MFCC,
    MFCC_HOP_LENGTH,MFCC_N_MELS,MFCC_FMAX,
    MFCC_FMIN,MFCC_WINDOW,
    MEL_N_MELS,MEL_FMAX,MEL_FMIN,MEL_HOP_LENGTH,MEL_N_FFT,MEL_WINDOW

)
#----------------------------------MFCC Feature Extraction-------------------------------------#
def extract_mfcc(audio:np.ndarray,
                         sr:int=SAMPLE_RATE,
                         n_mfcc:int=MFCC_N_MFCC,
                         n_fft:int = MFCC_N_FFT,
                         hop_length:int=MFCC_HOP_LENGTH,
                         n_mels:int=MFCC_N_MELS,
                         window:str=MFCC_WINDOW,
                         fmin:float=MFCC_FMIN,
                         fmax:float=MFCC_FMAX):
    '''
      MFCC feature extraction of raw audio form in numpy array 
      returns an np.ndarray of shape (n_mfcc,time frames)
    '''
    mfcc = librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,window=window,fmin=fmin,fmax=fmax)

    # Normalize the MFCC feature array
    mfcc = (mfcc - mfcc.mean(axis=1,keepdims=True))/(mfcc.std(axis=1,keepdims=True)+ 1e-8)
    return mfcc.astype(np.float32)

def extract_mfcc_delta(audio:np.ndarray,sr:int=SAMPLE_RATE):
    '''
    This function is choosen extract the delta and delta-delta MFCC features 
    Base-MFCC -> Static spectral audio/ spectral details of frames individually.
    delta-MFCC -> the rate of change of MFCC over time
    delta-delta-MFCC -> the acceleration/change of the delta-MFCC over time
    returns (3(three channels/feature types), n_mfcc, time_frames)
    '''
    mfcc = extract_mfcc(audio,sr=sr)
    delta = librosa.feature.delta(mfcc,order=1)
    delta_delta = librosa.feature.delta(delta,order=2)
    return np.stack([mfcc,delta,delta_delta],axis=0) # Multi channel Tensor for Model Training

#-----------------------------------------Log-Mel Feature Extraction------------------------------------------#

def extract_log_mel(audio:np.ndarray,
                    sr:int=SAMPLE_RATE,
                    n_fft:int=MEL_N_FFT,
                    hop_length:int=MEL_HOP_LENGTH,
                    n_mels:int=MEL_N_MELS,
                    fmin:float=MEL_FMIN,
                    fmax:float=MEL_FMAX,
                    window:str=MEl_WINDOW):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax,window=window)
    log_mel = librosa.power_to_db(S=mel_spectrogram,ref=np.max)
    #Normalizing the features
    # we are doing min-max normalization for the log-mel features
    log_mel = (log_mel - log_mel.min())/(log_mel.max()-log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)

def extract_log_mel_channel(audio:np.ndarray,sr:int=SAMPLE_RATE):
    # Created this function just to make the CNN input tensor same dimension to the MFCC one
    log_mel=extract_log_mel(audio=audio,sr=sr)
    return log_mel[np.newaxis, ... ]# this returns (1 , n_mels ,time frames)

def extract_features(feature_type:str,audio:np.ndarray,sr:int=SAMPLE_RATE):
    feature_type = feature_type.lower()
    if feature_type == "mfcc":
        feat = extract_mfcc(audio, sr=sr)   # (n_mfcc, T)
        return feat[np.newaxis, ...]         # (1, n_mfcc, T)
    elif feature_type in ("logmel", "log_mel", "mel"):
        return extract_log_mel_channel(audio, sr=sr)  # (1, n_mels, T)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}. Choose 'mfcc' or 'logmel'.")

def get_feature_shape(feature_type: str = "mfcc",
                      max_duration: float = 4.0,
                      sr: int = SAMPLE_RATE) -> tuple:
    """Returns expected (C, H, W) shape for CNN input."""
    dummy = np.zeros(int(max_duration * sr))
    feat  = extract_features(dummy, feature_type=feature_type, sr=sr)
    return feat.shape




if __name__ == "__main__":
    print("Testing feature extraction...")
    sr = 16000
    audio = np.random.randn(4 * sr).astype(np.float32)

    mfcc   = extract_features(audio = audio, feature_type="mfcc")
    logmel = extract_features(audio = audio, feature_type="logmel")

    print(f"  MFCC shape  : {mfcc.shape}   dtype={mfcc.dtype}")
    print(f"  LogMel shape: {logmel.shape}  dtype={logmel.dtype}")
    print(f"  MFCC  range : [{mfcc.min():.3f}, {mfcc.max():.3f}]")
    print(f"  LogMel range: [{logmel.min():.3f}, {logmel.max():.3f}]")
    print("Feature extraction OK.")
    


    

    


