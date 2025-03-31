import os

class Config:
    DATA_DIR = "data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    MODEL_DIR = "models"
    FEATURE_DIR = "features"
  
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 5  
    N_FFT = 2048
    HOP_LENGTH = 512
    
    USE_OPEN_SMILE = False  
    USE_WORLD = True
    USE_HPSS = True  
   
    BATCH_SIZE = 128
    EPOCHS = 150  
    LEARNING_RATE = 0.0001
    EARLY_STOPPING_PATIENCE = 15
    CLASS_WEIGHTS = None  
    
    NUM_WORKERS = max(1, os.cpu_count() - 1)
    RANDOM_SEED = 42

config = Config()