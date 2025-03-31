import tensorflow as tf
import numpy as np
from features import process_dataset
from config import config
import os
import pickle
from sklearn.utils.class_weight import compute_class_weight

def load_features(split):
    features, labels = [], []
    for label, val in [('real', 0), ('fake', 1)]:
        with open(os.path.join(config.FEATURE_DIR, f"{split}_{label}.pkl"), 'rb') as f:
            data = pickle.load(f)
            features.append(data['features'])
            labels.extend([val] * len(data['features']))
    return np.vstack(features), np.array(labels)

def create_enhanced_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    
    x = tf.keras.layers.BatchNormalization()(inputs)
    
    # Deep layers with swish activation
    x = tf.keras.layers.Dense(512, activation='swish')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(256, activation='swish')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid',
                                  bias_initializer=tf.keras.initializers.Constant(-0.5))(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def main():
    if not all(os.path.exists(os.path.join(config.FEATURE_DIR, f"{s}_{l}.pkl")) 
               for s in ['train', 'val'] for l in ['real', 'fake']):
        print("Generating features...")
        process_dataset('train')
        process_dataset('val')
    
   
    X_train, y_train = load_features('train')
    X_val, y_val = load_features('val')
    
    config.CLASS_WEIGHTS = dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    ))
  
    model = create_enhanced_model(X_train.shape[1])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        class_weight=config.CLASS_WEIGHTS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(config.MODEL_DIR, 'best_model.h5'),
                save_best_only=True,
                monitor='val_auc',
                mode='max'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
    )
    
    model.save(os.path.join(config.MODEL_DIR, 'final_model.h5'))

if __name__ == "__main__":
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    main()