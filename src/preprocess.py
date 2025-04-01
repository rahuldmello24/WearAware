import os
import pandas as pd
import numpy as np
from glob import glob

# CONFIG
DATA_DIR = "/Users/rahuldmello/Documents/Projects/WearAware/data/emowear_raw_data"
OUTPUT_PATH = "../data/processed/X.npy"
TARGET_FREQ = 4
WINDOW_SIZE_SEC = 60
STEP_SIZE_SEC = 30

SIGNAL_FILES = [
    "signals-bh3-acc.csv", "signals-bh3-bb.csv", "signals-bh3-br.csv", "signals-bh3-ecg.csv", "signals-bh3-hr.csv",
    "signals-bh3-hr_confidence.csv", "signals-bh3-rr.csv", "signals-bh3-rsp.csv", "signals-e4-acc.csv",
    "signals-e4-bvp.csv", "signals-e4-eda.csv", "signals-e4-hr.csv", "signals-e4-ibi.csv", "signals-e4-skt.csv"
]

def load_signals(subject_folder):
    """
    Load and align all sensor signal files from a single subject folder.
    Handles CSVs with 2+ columns. Returns merged DataFrame
    """

    dfs = []
    for file in SIGNAL_FILES:
        path = os.path.join(subject_folder, file)
        if not os.path.exists(path):
            continue # skip missing files
        df = pd.read_csv(path)

        # rename all columns with unique identifiers
        new_cols = ['timestamp'] + [f"{file.replace('.csv', '')}_{i}" for i in range(1, df.shape[1])]
        df.columns = new_cols
        dfs.append(df)

    if not dfs:
        return None

    # merge dataframes on timestamp using nearest match
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge_asof(merged.sort_values("timestamp"),
                               df.sort_values("timestamp"),
                               on="timestamp", direction="nearest")

    # drop any rows with missing values after merge
    merged = merged.dropna()
    return merged


def resample_signals(df, freq):
    """
    Resample signals to a uniform frequency using interpolation
    """
    start = df['timestamp'].min()
    end = df['timestamp'].max()
    target_ts = np.arange(start, end, 1 / freq) # uniform time axis
    resampled = pd.DataFrame({"timestamp": target_ts})
    for col in df.columns:
        if col == "timestamp": continue
        # interpolate each signal to align with new time axis
        resampled[col] = np.interp(target_ts, df['timestamp'], df[col])
    return resampled


def normalize_signals(df):
    """
    Normalize each signal to mean=0 and std=1 (z-score normalization)
    """
    for col in df.columns:
        if col != "timestamp":
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def slice_windows(df, window_sec, step_sec, freq):
    """
    Slice the time-series into overlapping windows
    Each window is of shape (window_len, num_signals)
    """
    window_len = int(window_sec * freq)
    step_len = int(step_sec * freq)
    windows = []
    for start in range(0, len(df) - window_len + 1, step_len):
        # drop timestamp before slicing into model-ready matrix
        window = df.iloc[start:start + window_len].drop(columns=['timestamp']).values
        windows.append(window)
    return np.stack(windows) if windows else None


# MAIN EXECUTION
all_windows = []

# get a list of all subject folders
subject_folders = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]
print(f"Found {len(subject_folders)} subjects")

# loop through each subject folder
for folder in subject_folders:
    print(f"Processing {os.path.basename(folder)}")

    # load and align signals
    df = load_signals(folder)
    if df is None:
        print("No valid signal data, skipping")
        continue

    # resample to uniform frequency
    df = resample_signals(df, TARGET_FREQ)

    # normalize signals
    df = normalize_signals(df)

    # slice into windows
    windows = slice_windows(df, WINDOW_SIZE_SEC, STEP_SIZE_SEC, TARGET_FREQ)
    if windows is not None:
        all_windows.append(windows)


# SAVE AND CONCAT
# combine all subjects into a single numpy array
X = np.concatenate(all_windows, axis=0)

# ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# save dataset
np.save(OUTPUT_PATH, X)
print(f"\n Done! Final Dataset shape: {X.shape} saved to {OUTPUT_PATH}")