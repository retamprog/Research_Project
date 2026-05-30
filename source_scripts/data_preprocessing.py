# data_preprocessing python script
'''
 Audio Preprocessing functions Py script
 1> Audio loading by librosa ->
 2> Silence triming of audio
 3> Pad/truncate audio to uniform duration
 4> Normalizing audio
 5> Preprocessing pipeline
 6> Preprocessing in batch

 the preprocessing pipeline which will be defined in the preprocessing_audio function

 Preprocessing pipeline:
 loading audio using librosa -> Triming silence --> padding/truncating audio to fixed duration

 
 
'''
import os
import pandas as pd
import numpy as np
import librosa
import sys
# using the sys module to add the parent directory path to the import file search options which is the sys.path.
# this will enable to access the config.py functions in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from config import SAMPLE_RATE,MAX_DURATION,TRIM_SILENCE,FLAC_TRAIN_DIR

def load_audio(filepath:str,sr:int = SAMPLE_RATE):
    # loading and resampling of audio to target sampling rate
    try:
        audio,orig_sr=librosa.load(filepath,sr=sr)
        return audio
    except Exception as e:
        raise IOError(f"Could not find audio file {filepath}: {e}")

def pad_truncate(audio: np.ndarray,max_duration:float=MAX_DURATION,sr:int=SAMPLE_RATE):
    # pad and truncate the audio files to get uniform audio duration
    max_length=int(max_duration*sr) 
    #librosa.util.fix_length(audio,size=fixed_length)
    if len(audio) > max_length:
        audio=audio[:max_length]
    else:
        pad_len=max_length-len(audio)
        audio=np.pad(audio,pad_width=(0,pad_len),mode='constant',constant_values=0)    ## Error line 

    return audio    

def trim_silence(audio:np.ndarray,top_db:int=30):
    audio,index=librosa.effects.trim(y=audio,top_db=top_db)
    return audio

# Not doing peak normalization of the audio may cause loss in important features due to amplitude variances 

def preprocess_audio(filepath:str,sr:int=SAMPLE_RATE,max_duration:float=MAX_DURATION,trim_silen:bool=TRIM_SILENCE):
    # single audio file preprocessing function 
    # the single wav file audio loading and resampling.
    audio=load_audio(filepath,sr=sr)
    if trim_silen:
        audio=trim_silence(audio)
    
    audio=pad_truncate(audio,max_duration=max_duration,sr=sr)

    return audio

def preprocess_batch(filepaths:list,**kwargs):
    # Multiple audio file preprocessing in batches
    results=[]
    for fp in filepaths:
        # storing the results in the format (filepath,audio,error)
        try:
            audio=preprocess_audio(fp,**kwargs)
            results.append((fp,audio,None))
        except Exception as e:
            results.append((fp,None,e))

    return results

# test script for checking the functions are individually working or not
if __name__=='__main__':
    test_audio_path=os.path.join(FLAC_TRAIN_DIR,'LA_T_1000137.flac')
    # testing audio load and resampling
    sr=SAMPLE_RATE
    t_audio=load_audio(test_audio_path,sr)
    print(f'The resampled audio at sampling rate of {sr} : {t_audio}')
    
    # trim silence before and after the original audio file
    # t_audio_silen,index=trim_silence(t_audio,top_db=30)
    # # here the index = (start,end) stores the sample indexes from where the non-silent part starts and ends.
    # silence_before=(index[0]-t_audio[0])/sr
    # silence_after=(t_audio[len(t_audio)-1]-index[1])/sr
    # print(f'the duration of silence before {silence_before} and the duration of silence after {silence_after}')

    ## testing audio preprocessing 
    audio  = preprocess_audio(filepath=test_audio_path,sr=SAMPLE_RATE)


        





    



