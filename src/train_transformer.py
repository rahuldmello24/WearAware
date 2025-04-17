import tensorflow as tf
import numpy as np
import keras
from keras import layers, models, Input, callbacks
from sklearn.model_selection import train_test_split

def build_transformer_model(input_shape, latent_dim):
    inputs = Input(shape=input_shape)

    # positional encoding layer 
    x = layers.Dense(latent_dim)(inputs) # ensure embedding dimension matches latent_dim

    # transformer encoder block
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=latent_dim)(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)

    ffn_output = layers.Dense(128, activation='relu')(x)
    ffn_output = layers.Dense(latent_dim)(ffn_output)
    x = layers.Add()([x, ffn_output])
    x = layers.LayerNormalization()(x)

    # global average pooling
    x = layers.GlobalAveragePooling1D()(x)

    # final prediction layer
    outputs = layers.Dense(latent_dim)(x)

    model = models.Model(inputs, outputs, name='transformer_latent_predictor')
    return model


# load sequences
X_seq = np.load('data/processed/latent_sequences/X_seq.npy')
y_seq = np.load('data/processed/latent_sequences/y_seq.npy')

# split
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# build model
latent_dim = X_seq.shape[2]
model = build_transformer_model(input_shape=X_seq.shape[1:], latent_dim=latent_dim)

model.compile(optimizer='adam', loss='mse')
model.summary()


# train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs = 50,
    batch_size = 64,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# save
model.save('models/transformers/transformer_latent_predictor_1.keras')
