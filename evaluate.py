import numpy as np
import tensorflow as tf
from config import config
import os
import pickle
from sklearn.metrics import (classification_report, 
                           roc_auc_score, 
                           confusion_matrix,
                           average_precision_score)
from sklearn.metrics import roc_curve

def load_features(split):
    features, labels = [], []
    for label, val in [('real', 0), ('fake', 1)]:
        pkl_path = os.path.join(config.FEATURE_DIR, f"{split}_{label}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Feature file not found: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            features.append(data['features'])
            labels.extend([val] * len(data['features']))
    
    return np.vstack(features), np.array(labels)

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold

def main():
    required_files = [f"test_{label}.pkl" for label in ['real', 'fake']]
    if not all(os.path.exists(os.path.join(config.FEATURE_DIR, f)) for f in required_files):
        print("Test features not found. Running feature extraction...")
        from features import process_dataset
        process_dataset('test')
    
    try:
        model = tf.keras.models.load_model(os.path.join(config.MODEL_DIR, 'best_model.h5'))
        X_test, y_test = load_features('test')
        
        y_pred = model.predict(X_test).flatten()
        y_pred_class = (y_pred > 0.4).astype(int)  
        

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_class, 
                                  target_names=['real', 'fake'],
                                  digits=4))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_class))
        
        print("\nROC AUC Score:", roc_auc_score(y_test, y_pred))
        print("Average Precision:", average_precision_score(y_test, y_pred))
        
        eer, threshold = calculate_eer(y_test, y_pred)
        print(f"\nEqual Error Rate: {eer:.4f} (threshold: {threshold:.4f})")
        
    except Exception as e:
        print(f"\nEvaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()