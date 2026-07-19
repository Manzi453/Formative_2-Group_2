"""Shared feature-extraction helpers, reused by the pipeline notebooks and the CLI app,
so a raw image/audio file is always turned into a feature vector the exact same way
whether it happened during training or during a live authentication attempt.
"""
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
import librosa

# ---- image ----
def load_image(path):
    return Image.open(path).convert('RGB')

def color_histogram(img, bins=16):
    arr = np.array(img)
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 255), density=True)
        feats.extend(hist.tolist())
    return feats

def hog_features(img, pixels_per_cell=(16, 16)):
    gray = rgb2gray(np.array(img))
    fd = hog(gray, orientations=9, pixels_per_cell=pixels_per_cell, cells_per_block=(2, 2), feature_vector=True)
    hog_hist, _ = np.histogram(fd, bins=32, density=True)
    return hog_hist.tolist()

def grayscale_stats(img):
    gray = np.array(img.convert('L'), dtype=np.float64)
    return {'gray_mean': gray.mean(), 'gray_std': gray.std(), 'gray_min': gray.min(), 'gray_max': gray.max()}

def extract_image_feature_vector(path, feature_cols):
    """Loads an image file and returns a feature vector ordered to match `feature_cols`
    (the column order the model was trained on)."""
    img = load_image(path)
    row = {}
    row.update(grayscale_stats(img))
    for i, v in enumerate(color_histogram(img)):
        row[f'color_hist_{i}'] = v
    for i, v in enumerate(hog_features(img)):
        row[f'hog_hist_{i}'] = v
    return np.array([row[c] for c in feature_cols])

# ---- audio ----
def load_audio(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def extract_audio_feature_vector(path, feature_cols):
    """Loads an audio file and returns a feature vector ordered to match `feature_cols`."""
    y, sr = load_audio(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    row = {}
    for i in range(mfcc.shape[0]):
        row[f'mfcc_{i}_mean'] = mfcc[i].mean()
        row[f'mfcc_{i}_std'] = mfcc[i].std()
    row['spectral_rolloff_mean'] = rolloff.mean()
    row['spectral_rolloff_std'] = rolloff.std()
    row['rms_energy_mean'] = rms.mean()
    row['rms_energy_std'] = rms.std()
    row['duration_sec'] = len(y) / sr
    return np.array([row[c] for c in feature_cols])
