# Python script for loading the data for Pytorch model 

import pandas as pd
import sys
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source_scripts.data_preprocessing import preprocess_audio,pad_truncate
from source_scripts.feature_extraction import extract_features
from utility_scripts.data_augment import augment_audio

from config import (
    TRAIN_PROTOCOL,DEV_PROTOCOL,EVAL_PROTOCOL,
    FLAC_DEV_DIR,FLAC_EVAL_DIR,FLAC_TRAIN_DIR,
    NUM_WORKERS,SAMPLE_RATE,BATCH_SIZE
)

def protocol_loader(path):
    data =[]
    with open(path,'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # speaker_id,Audio_file_name,-,Systemid,key
            speaker_id=parts[0]
            audio_file_id=parts[1]
            attack_id=parts[3]
            label_str=parts[4]
            label=0 if label_str=='bonafide' else 1

            data.append({"speaker_id":speaker_id,
                          "audio_id":audio_file_id,
                          "attack_id":attack_id,
                          "label_str":label_str,
                          "label":label})
    
    return pd.DataFrame(data)

# creating a ASVspoof dataset loader class based on the Torch abstract class Dataset
# An abstract class representing a Dataset.

#All datasets that represent a map from keys to data samples should subclass it. 
# All subclasses should overwrite __getitem__(), supporting fetching a data sample for a given key. 
# Subclasses could also optionally overwrite __len__(), which is expected to return the size of the dataset 
# by many Sampler implementations and the default options of DataLoader. 
# Subclasses could also optionally implement __getitems__(), for speedup batched samples loading. 
# This method accepts list of indices of samples of batch and returns list of samples.
class ASVspoofDataset(Dataset):

    '''
    ASVspoof 2019 LA dataset loader class 
    Input:
    audio_dir,protocol_path,augment_param,cache and sample rate
    Output
    Returns Feature Tensor of shape (1 or 3,freq_bins,time_frames)
            label tensor scalar (0=bonafide,1=Spoof)
            filepath: str - path to the original audio file
    ''' 
   #creating the constructor for class 
    def __init__(self,audio_dir,
                 protocol_path,
                 feature_type:str='mfcc',
                 augment:bool=False,
                 cache_allowed:bool = False,
                 sr:int=SAMPLE_RATE):
        
        self.audio_dir=audio_dir
        # self.protocol_dir=protocol_dir the protocol_path var later not needed
        self.feature_type=feature_type
        self.augment=augment
        self.sr=sr
        self.cache={} if cache_allowed else None # this is for caching used features uses RAM but useful
        # the cache stores the filepath (key) and the feature_tensor (value)
        self.df = protocol_loader(protocol_path)
        print(f"Dataset of length {len(self.df)} loaded from {os.path.basename(protocol_path)}")
        print(f"Bonafide samples :  {(self.df['label']==0).sum()} Spoof samples : {(self.df['label']==1).sum()}")



    def __getitem__(self, index):
        row = self.df.iloc[index]
        audio_id = row['audio_id']
        filepath =  os.path.join(self.audio_dir,audio_id+'.flac')
        if self.cache is not None and filepath in self.cache:
            self.cache[filepath] = feature_np   # this will return the tensor for the particular audio file if stored in cache
        else:
            audio = preprocess_audio(filepath,sr=self.sr)
            if self.augment:
                audio = augment_audio(audio)
                # forcing fixed length even after augmentation
                audio = pad_truncate(audio,sr=self.sr)

            feature_np = extract_features(feature_type=self.feature_type,audio=audio,sr=self.sr)
            # feature_tensor = torch.from_numpy(feature_np)
            if self.cache is not None:
                self.cache[audio]=feature_np
        # print(len(audio))        
        # print(feature_np.shape)        
        feature_tensor = torch.from_numpy(feature_np)
        label_tensor = torch.tensor(row['label'],dtype= torch.long)
        return feature_tensor,label_tensor,filepath
            
    
    def __len__(self):
       return len(self.df)

def get_dataloaders(feature_type: str = "mfcc",
                    batch_size: int = BATCH_SIZE,
                    augment_train: bool = True,
                    num_workers: int = NUM_WORKERS):
    """
    Create train / dev / eval DataLoaders for ASVspoof 2019 LA.
    """
    train_ds = ASVspoofDataset(FLAC_TRAIN_DIR, TRAIN_PROTOCOL, feature_type=feature_type, augment=augment_train)
    dev_ds   = ASVspoofDataset(FLAC_DEV_DIR,   DEV_PROTOCOL,   feature_type=feature_type, augment=False)
    eval_ds  = ASVspoofDataset(FLAC_EVAL_DIR,  EVAL_PROTOCOL,  feature_type=feature_type, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, dev_loader, eval_loader    

            






    # def __getitem__(self, index):
        

if __name__ == "__main__":
    print("Testing with real dataset...")
    # ds = SyntheticAudioDataset(num_samples=200, feature_type="mfcc")
    ds = ASVspoofDataset(audio_dir=FLAC_TRAIN_DIR,protocol_path=TRAIN_PROTOCOL,feature_type="mfcc",sr=SAMPLE_RATE)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    for feat, label, _ in dl:
        print(f"  Batch — features: {feat.shape}, labels: {label.shape}")
        break
    print("Dataset OK.")









