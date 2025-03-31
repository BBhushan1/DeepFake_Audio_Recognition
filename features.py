import librosa
import numpy as np
import pyworld as pw
import opensmile
from config import config
import os
import pickle
from tqdm import tqdm

class EnhancedFeatureExtractor:
    def __init__(self):
        self.feature_types = []  
        self.smile = None
        if config.USE_OPEN_SMILE:
            try:
                self.smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals,
                )
                self.feature_types.append(('opensmile', 88))
            except Exception as e:
                print(f"OpenSMILE disabled: {e}")

    def _extract_spectral_features(self, y, sr):
        features = {}
        stft = librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        stft_mag = np.abs(stft)
        
        # Core spectral features
        spectral = {
            'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40),
            'chroma': librosa.feature.chroma_stft(S=stft_mag, sr=sr),
            'contrast': librosa.feature.spectral_contrast(S=stft_mag, sr=sr),
            'tonnetz': librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr),
            'rms': librosa.feature.rms(y=y),
            'zcr': librosa.feature.zero_crossing_rate(y),
            'rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr)
        }
        
        # HPSS features if enabled
        if config.USE_HPSS:
            y_harm, y_perc = librosa.effects.hpss(y)
            spectral.update({
                'harm_zcr': librosa.feature.zero_crossing_rate(y_harm),
                'perc_rms': librosa.feature.rms(y=y_perc)
            })
        
        # Process all spectral features
        for name, feat in spectral.items():
            features[name] = feat.mean(axis=1) if len(feat.shape) > 1 else feat
            self.feature_types.append((name, features[name].shape[0]))
        
        return features
    
    def _extract_vocal_features(self, y, sr):
        features = {}
        if config.USE_WORLD:
            try:
                y_world = y.astype(np.float64)
                f0, time_axis = pw.harvest(y_world, sr)
                sp = pw.cheaptrick(y_world, f0, time_axis, sr)
                ap = pw.d4c(y_world, f0, time_axis, sr)
                
                valid_f0 = f0[f0 > 0]
                features.update({
                    'f0_mean': valid_f0.mean() if len(valid_f0) > 0 else 0.0,
                    'f0_std': valid_f0.std() if len(valid_f0) > 0 else 0.0,
                    'sp_mean': sp.mean(axis=1),
                    'ap_mean': ap.mean(axis=1)
                })
                self.feature_types.extend([
                    ('f0', 2), ('sp', 60), ('ap', 60)
                ])
            except Exception as e:
                print(f"WORLD features disabled: {e}")
        return features
    
    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, duration=config.MAX_AUDIO_LENGTH)
            if len(y) < 1024:
                raise ValueError("Audio too short")
            
            features = {}
            features.update(self._extract_spectral_features(y, sr))
            features.update(self._extract_vocal_features(y, sr))
            
            if self.smile:
                try:
                    smile_feats = self.smile.process_file(audio_path).values.flatten()
                    features['opensmile'] = smile_feats
                except Exception as e:
                    print(f"OpenSMILE failed: {e}")
            
            # Convert all to numpy arrays and concatenate
            return np.concatenate([np.array(v).flatten() for v in features.values()])
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

def process_dataset(dataset_type):
    extractor = EnhancedFeatureExtractor()
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    
    for label in ['real', 'fake']:
        label_dir = os.path.join(getattr(config, f"{dataset_type.upper()}_DIR"), label)
        if not os.path.exists(label_dir):
            continue
            
        audio_files = [f for f in os.listdir(label_dir) if f.endswith(('.wav', '.flac', '.mp3'))]
        features, valid_files = [], []
        
        for file in tqdm(audio_files, desc=f"{dataset_type}/{label}"):
            audio_path = os.path.join(label_dir, file)
            feat = extractor.extract_features(audio_path)
            if feat is not None:
                features.append(feat)
                valid_files.append(file)
        
        # Save features with metadata
        output_path = os.path.join(config.FEATURE_DIR, f"{dataset_type}_{label}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump({
                'files': valid_files,
                'features': np.array(features),
                'feature_types': extractor.feature_types
            }, f)