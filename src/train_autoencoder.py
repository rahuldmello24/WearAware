import numpy as np
import tensorflow as tf
from keras import layers, models, Input
import os

# Load data
X = np.load('data/processed/X.npy')

# check data shape
print("Input shape:", X.shape)


# Build Autoencoder
def build_autoencoder(input_shape=(240, 18), latent_dim=256):
    inputs = Input(shape=input_shape)

    # encoder
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2, padding='same')(x) # 120
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x) # 60
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x) # 30
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent_vector')(x)

    # decoder 
    x = layers.Dense(30 * 128)(latent)
    x = layers.Reshape((30, 128))(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x) # 60
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x) # 120
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x) # 240
        
    outputs = layers.Conv1D(input_shape[1], 3, activation='linear', padding='same')(x)

    autoencoder = models.Model(inputs, outputs, name='autoencoder')
    encoder = models.Model(inputs, latent, name='encoder')

    return autoencoder, encoder

# instantiate models
autoencoder, encoder = build_autoencoder()

# training
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

history = autoencoder.fit(
    X_train, X_train,
    epochs=75,
    batch_size=64,
    validation_data=(X_val, X_val) 
)

# save models
os.makedirs('models', exist_ok=True)
autoencoder.save('models/AEs/autoencoder_full_conv.keras')
encoder.save('models/AEs/encoder_only_conv.keras')

print('Training complete.')