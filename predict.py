import numpy as np
import librosa
import tensorflow as tf
from config import config
import os
import sys
from features import EnhancedFeatureExtractor

class AudioDeepfakeDetector:
    def __init__(self, model_path=None):

        self.model_path = model_path or os.path.join(config.MODEL_DIR, 'best_model.h5')
        self.extractor = EnhancedFeatureExtractor()
        self.model = self._load_model()
        self.threshold = 0.5
    
    def _load_model(self):
        try:
            return tf.keras.models.load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
            sys.exit(1)
    
    def _extract_features(self, audio_path):
        try:
            # Validate file existence
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Extract features
            features = self.extractor.extract_features(audio_path)
            if features is None:
                raise ValueError("Feature extraction failed")
                
            expected_dim = self.model.input_shape[1]
            if len(features) != expected_dim:
                raise ValueError(
                    f"Feature dimension mismatch. Expected {expected_dim}, got {len(features)}"
                )
                
            return features
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def predict_proba(self, audio_path):
        features = self._extract_features(audio_path)
        if features is None:
            return None
            
        return float(self.model.predict(np.expand_dims(features, axis=0))[0][0])
    
    def predict(self, audio_path, threshold=None):
        threshold = threshold or self.threshold
        result = {
            'prediction': None,
            'confidence': None,
            'probability': None,
            'threshold': threshold,
            'status': 'error',
            'error': None
        }
        
        try:
            proba = self.predict_proba(audio_path)
            if proba is None:
                raise ValueError("Prediction failed")
                
            result.update({
                'probability': proba,
                'prediction': 'fake' if proba > threshold else 'real',
                'confidence': proba if proba > threshold else 1 - proba,
                'status': 'success',
                'error': None
            })
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def predict_batch(self, audio_paths, threshold=None):
        features = []
        valid_paths = []
        
        for path in audio_paths:
            feat = self._extract_features(path)
            if feat is not None:
                features.append(feat)
                valid_paths.append(path)
        
        if not features:
            return []
            
        # Batch prediction
        probabilities = self.model.predict(np.array(features)).flatten()
        threshold = threshold or self.threshold
        
        return [
            {
                'file': path,
                'prediction': 'fake' if proba > threshold else 'real',
                'probability': float(proba),
                'confidence': proba if proba > threshold else 1 - proba,
                'threshold': threshold
            }
            for path, proba in zip(valid_paths, probabilities)
        ]

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_path> [threshold]")
        print("Example: python predict.py test.wav 0.35")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None
    
    detector = AudioDeepfakeDetector()
    result = detector.predict(audio_path, threshold)
    
    if result['status'] == 'success':
        print("\nDeepfake Detection Result:")
        print(f"File: {audio_path}")
        print(f"Prediction: {result['prediction']} (threshold: {result['threshold']:.2f})")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Raw Probability: {result['probability']:.4f}")
    else:
        print(f"\nPrediction failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()