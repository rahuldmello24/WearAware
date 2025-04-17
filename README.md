# WearAware: Unsupervised Anomaly Detection from Wearable Sensor Signals

This project explores **unsupervised anomaly detection** from **multimodal wearable sensor data**, using a combination of **Autoencoders**, **Transformers**, and **signal analysis** techniques. It includes preprocessing pipelines, latent representation learning, clustering, and rich visualization of anomalies via signal and frequency domain statistics.

---

## Project Structure

```
.
├── data/
│   ├── emowear_raw_data/             # Original raw sensor CSVs
│   └── processed/                    # Processed data
│       ├── X.npy                     # Main preprocessed input signal windows
│       ├── Z_256.npy                 # Latents from AE
│       ├── Z_autoencoder1.npy        # Alternate AE latents
│       ├── anomaly_clusters.npy      # Cluster labels
│       ├── anomaly_indices.npy       # High-error sample indices
│       └── transformer_1_errors.npy  # Transformer error scores
│
├── models/
│   ├── AEs/                          # Trained Autoencoders & Encoders
│   └── transformers/                # Trained Transformers
│
├── notebooks/
│   ├── autoencoder_analysis.ipynb    # AE training & latent exploration
│   ├── transformer_analysis.ipynb    # Transformer error analysis
│   ├── anomaly_clustering_signal_backtrack.ipynb # Signal-level analysis of clusters
│   ├── raw_signal_anomaly_analysis.ipynb         # Backtrack high-error samples
│   └── anomaly_viz.ipynb            # Visualization of sensor stats and FFT
│
├── src/
│   ├── preprocess.py                 # Raw signal preprocessing
│   ├── preprocess_latents.py        # Generate AE latents
│   ├── train_autoencoder.py         # AE training
│   └── train_transformer.py         # Transformer training
│
└── results/                          # Output folder for plots & exports (optional)
```

---

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess data:**
   ```bash
   python src/preprocess.py
   ```

3. **Train Autoencoder and extract latent features:**
   ```bash
   python src/train_autoencoder.py
   python src/preprocess_latents.py
   ```

4. **Train Transformer on AE latents:**
   ```bash
   python src/train_transformer.py
   ```

5. **Run analysis notebooks:**
   Open and explore:
   - `notebooks/autoencoder_analysis.ipynb`
   - `notebooks/anomaly_clustering_signal_backtrack.ipynb`
   - `notebooks/anomaly_viz.ipynb`

---

## Core Features

- Signal preprocessing (windowing, normalization)
- Autoencoder latent representation learning
- Transformer-based reconstruction error analysis
- KMeans clustering of latent anomalies
- In-depth raw signal backtracking
- Sensor-level and FFT domain visualization
- Exploratory analysis of physiological states

---

## Notes

- The raw signal files are not included in the repo. You may need to populate `emowear_raw_data/` with source `.csv` files.
- You can download the raw '.csv' files from here: https://zenodo.org/records/10407279

---

## Acknowledgements

This project was inspired by work in:
- Unsupervised anomaly detection
- Physiological signal modeling
- Self-supervised representation learning in time-series
