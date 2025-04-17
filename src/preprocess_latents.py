import numpy as np
import os

# PATHS
DATA_PATH = 'data/processed/Z_256.npy'
OUTPUT_DIR = 'data/processed/latent_sequences'
SEQUENCE_LEN = 10 # number of windows per sequence

# load latent vectors
print("Loading Data...")
Z = np.load(DATA_PATH)

# create overlapping sequences
X_seq = np.stack([Z[i:i+SEQUENCE_LEN] for i in range(len(Z) - SEQUENCE_LEN)])
y_seq = Z[SEQUENCE_LEN:] # target: next latent vector

# create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# save sequences
np.save(os.path.join(OUTPUT_DIR, 'X_seq.npy'), X_seq)
np.save(os.path.join(OUTPUT_DIR, 'y_seq.npy'), y_seq)

# info
print('Latent sequence chunking complete')
print(f'X_seq shape: {X_seq.shape}')
print(f'y_seq shape: {y_seq.shape}')