import torch
# import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class Custom_cnn(nn.Module):
    def __init__(self,in_channels:int =1,
                 freq_bins:int =40,
                 num_classes:int =2,
                 dropout:float = 0.3):
        super(Custom_cnn,self).__init__()

        # the CNN model has two parts the feature extraction (Convulution and pooling layers) part and classifier (Dense layer) part
        # Feature extraction part
        self.features = nn.Sequential(
            
            # block 1
            nn.Conv2d(in_channels,32,kernel_size = 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.1),

            #block 2
            nn.Conv2d(32,64,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.1),

            #block 3
            nn.Conv2d(64,128,kernel_size = 3,padding =1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d((4,4))

        )
       # the Classifier part which makes the decision whether the audio is spoof or bonafide
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4,256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256,num_classes)        
            )
    def forward(self,x:torch.Tensor):
        x=self.features(x)
        x=self.classifier(x)
        return x
    def model_shape(self):
        print(self)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        

if __name__=='__main__':
    model = Custom_cnn(freq_bins=40)
    print(model)        
    

