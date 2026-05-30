'''
Config.py 
The centrally managed configuration script for initialization of :
 - Protocol file paths
 - Audio file paths
 - Training Hyperparameters
 - feature parameters both MFCC and Log Mel
 - Model types 
'''
import os
import librosa 

# file path configurations

# Base files for reaching the other files change due to google file structure difference

# BASE_DIR=os.path.dirname(os.path.abspath(__file__))
# DATA_DIR=os.path.join(BASE_DIR,"dataset","LA")
if os.path.exists("/content"):
    BASE_DIR = "/content"
    DATA_DIR = "/content/datasets/LA"
else:
    BASE_DIR=os.path.dirname(os.path.abspath(__file__))
    DATA_DIR=os.path.join(BASE_DIR,"dataset","LA")

# protocol files
PROTOCOL_DIR=os.path.join(DATA_DIR,'ASVspoof2019_LA_cm_protocols')
TRAIN_PROTOCOL=os.path.join(PROTOCOL_DIR,"ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTOCOL=os.path.join(PROTOCOL_DIR,"ASVspoof2019.LA.cm.dev.trl.txt")
EVAL_PROTOCOL=os.path.join(PROTOCOL_DIR,"ASVspoof2019.LA.cm.eval.trl.txt")

# audio files directory
FLAC_TRAIN_DIR=os.path.join(DATA_DIR,'ASVspoof2019_LA_train','flac')
FLAC_EVAL_DIR=os.path.join(DATA_DIR,'ASVspoof2019_LA_eval','flac')
FLAC_DEV_DIR=os.path.join(DATA_DIR,'ASVspoof2019_LA_dev','flac')
# output files directory
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
MODEL_SAVE_DIR  = os.path.join(OUTPUT_DIR, "saved_models")
RESULTS_DIR     = os.path.join(OUTPUT_DIR, "results")
# Data preprocessing parameters

SAMPLE_RATE=16000 # All audio will be resampled at 16khz rate
MAX_DURATION=4.0  # All audio will be clipped/padded to 4 secs
TRIM_SILENCE=True # Trim leading/trailing silence in audio

# Feature extraction parameters

# MFCC parameters

MFCC_N_MFCC = 40
MFCC_N_FFT = 512
MFCC_HOP_LENGTH = 160
MFCC_N_MELS = 80
MFCC_FMIN = 20
MFCC_FMAX = 8000
MFCC_WINDOW = 'hamming'

# Log Mel parameters

MEL_N_MELS = 80
MEL_N_FFT=512
MEL_HOP_LENGTH=160
MEL_FMIN = 20
MEL_FMAX = 8000
MEL_WINDOW = 'hamming'

# Training  Parameters 
BATCH_SIZE      = 64
NUM_EPOCHS      = 10
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 4
DEVICE          = "cuda"       

# Augmentation parameters
NOISE_FACTOR = 0.005
TIME_MASK = 20
FREQ_MASK = 10

if __name__ == '__main__':
    print(FLAC_TRAIN_DIR)







