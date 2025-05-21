import os
from pathlib import Path
import numpy as np
import librosa

def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.hstack((mfcc.mean(axis=1), mfcc.std(axis=1)))

def extract_chroma(audio: np.ndarray, sr: int):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    return np.hstack((chroma.mean(axis=1), chroma.std(axis=1)))


def extract_spectral_contrast(audio: np.ndarray, sr: int):
    spec = librosa.feature.spectral_contrast(y=audio, sr=sr)
    return np.hstack((spec.mean(axis=1), spec.std(axis=1)))

def extract_features_folder(folder: str, sr: int = 22050, clip_duration: float = 5.0, min_rms: float = 1e-4):
    """
    Iterate through processed WAVs in folder/category and extract features 
    
    returns X and y arrays
    """
    X, y = [], []
    # mapping categories to integer labels
    label_map = {cat: idx for idx, cat in enumerate(sorted(os.listdir(folder)))}
    for cat, idx in label_map.items():
        cat_dir = Path(folder) / cat
        for wav_path in cat_dir.glob('*.wav'):
            audio, _ = librosa.load(wav_path, sr=sr, duration=clip_duration)
            # compute RMS energy
            rms = np.sqrt(np.mean(audio**2))
            if rms < min_rms:
                # skip nearly silent clips
                continue
            # extract features
            feats = np.hstack([
                extract_mfcc(audio, sr),
                extract_chroma(audio, sr),
                extract_spectral_contrast(audio, sr)
            ])
            X.append(feats)
            y.append(idx)
    # convert to numpy arrays
    if X:
        return np.vstack(X), np.array(y)
    else:
        return np.empty((0, )), np.empty((0, ))
