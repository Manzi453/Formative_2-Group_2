#!/usr/bin/env python3
"""Regenerate task4 feature CSVs from the CURRENT images/ and audio/ files, then
retrain and save all three models — identical logic to task4_models.ipynb.

Run from inside task4_models_and_demo/:  python3 retrain.py

This exists because the committed image_features.csv / audio_features.csv / models
were generated from older image/audio files (e.g. emmanuel's photos were .jpg, now .png;
recordings were .mp4, now .wav). Re-running this makes the CLI accept the real members
and reject the UNAUTHORIZED_ATTEMPT probe.
"""
import os
import glob
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss

RANDOM_STATE = 42
SR = 22050
UNAUTHORIZED_LABEL = 'UNAUTHORIZED_ATTEMPT'
EXPRESSIONS = ['neutral', 'smiling', 'surprised']
PHRASES = ['phrase1', 'phrase2']
IMG_EXTS = ('jpg', 'jpeg', 'png')
AUDIO_EXTS = ('wav', 'mp3', 'flac', 'm4a', 'mp4')
os.makedirs('models', exist_ok=True)

# ---------------- image feature extraction (identical to feature_extraction.py) ----------------
def load_image(path):
    return Image.open(path).convert('RGB')

def color_histogram(img, bins=16):
    arr = np.array(img); feats = []
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 255), density=True)
        feats.extend(h.tolist())
    return feats

def hog_features(img, pixels_per_cell=(16, 16)):
    gray = rgb2gray(np.array(img))
    fd = hog(gray, orientations=9, pixels_per_cell=pixels_per_cell, cells_per_block=(2, 2), feature_vector=True)
    hh, _ = np.histogram(fd, bins=32, density=True)
    return hh.tolist()

def grayscale_stats(img):
    g = np.array(img.convert('L'), dtype=np.float64)
    return {'gray_mean': g.mean(), 'gray_std': g.std(), 'gray_min': g.min(), 'gray_max': g.max()}

def extract_image_row(img):
    row = {}
    row.update(grayscale_stats(img))
    for i, v in enumerate(color_histogram(img)):
        row[f'color_hist_{i}'] = v
    for i, v in enumerate(hog_features(img)):
        row[f'hog_hist_{i}'] = v
    return row

IMG_AUG = {
    'original': lambda im: im,
    'rotated': lambda im: im.rotate(15, expand=True, fillcolor=(255, 255, 255)),
    'flipped': lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    'grayscale': lambda im: im.convert('L').convert('RGB'),
}

def discover_image_members():
    members = {}
    for m in sorted(os.listdir('images')):
        mp = os.path.join('images', m)
        if not os.path.isdir(mp):
            continue
        found = {}
        for exp in EXPRESSIONS:
            for ext in IMG_EXTS:
                c = os.path.join(mp, f'{exp}.{ext}')
                if os.path.exists(c):
                    found[exp] = c; break
        if found:
            members[m] = found
    return members

def build_image_features():
    recs = []
    for member, files in discover_image_members().items():
        for exp, path in files.items():
            base = load_image(path)
            for aug, fn in IMG_AUG.items():
                im = fn(base)
                r = extract_image_row(im)
                r.update({'member': member, 'expression': exp, 'augmentation': aug,
                          'source_file': path, 'width': im.size[0], 'height': im.size[1]})
                recs.append(r)
    df = pd.DataFrame(recs)
    id_cols = ['member', 'expression', 'augmentation', 'source_file', 'width', 'height']
    feat_cols = [c for c in df.columns if c not in id_cols]
    return df[id_cols + feat_cols]

# ---------------- audio feature extraction ----------------
def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y, sr

def extract_audio_row(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    r = {}
    for i in range(mfcc.shape[0]):
        r[f'mfcc_{i}_mean'] = mfcc[i].mean(); r[f'mfcc_{i}_std'] = mfcc[i].std()
    r['spectral_rolloff_mean'] = rolloff.mean(); r['spectral_rolloff_std'] = rolloff.std()
    r['rms_energy_mean'] = rms.mean(); r['rms_energy_std'] = rms.std()
    r['duration_sec'] = len(y) / sr
    return r

AUDIO_AUG = {
    'original': lambda y, sr: y,
    'pitch_shifted': lambda y, sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=3),
    'time_stretched': lambda y, sr: librosa.effects.time_stretch(y, rate=1.2),
    'noisy': lambda y, sr: y + 0.02 * np.random.RandomState(RANDOM_STATE).randn(len(y)),
}

def discover_audio_members():
    members = {}
    for m in sorted(os.listdir('audio')):
        mp = os.path.join('audio', m)
        if not os.path.isdir(mp):
            continue
        found = {}
        for ph in PHRASES:
            for ext in AUDIO_EXTS:
                c = os.path.join(mp, f'{ph}.{ext}')
                if os.path.exists(c):
                    found[ph] = c; break
        if found:
            members[m] = found
    return members

def build_audio_features():
    recs = []
    for member, files in discover_audio_members().items():
        for ph, path in files.items():
            y, sr = load_audio(path)
            for aug, fn in AUDIO_AUG.items():
                ay = fn(y, sr)
                r = extract_audio_row(ay, sr)
                r.update({'member': member, 'phrase': ph, 'augmentation': aug, 'source_file': path})
                recs.append(r)
    df = pd.DataFrame(recs)
    id_cols = ['member', 'phrase', 'augmentation', 'source_file']
    feat_cols = [c for c in df.columns if c not in id_cols]
    return df[id_cols + feat_cols]

# ---------------- open-set training helper ----------------
def train_biometric(df, id_cols, out_path, test_size=0.3):
    pool = df[df['member'] != UNAUTHORIZED_LABEL].copy()
    feat_cols = [c for c in df.columns if c not in id_cols]
    le = LabelEncoder()
    X = pool[feat_cols].values
    y = le.fit_transform(pool['member'])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    yp = model.predict(Xte); ypr = model.predict_proba(Xte)
    acc = accuracy_score(yte, yp)
    f1 = f1_score(yte, yp, average='macro')
    ll = log_loss(yte, ypr, labels=list(range(len(le.classes_))))
    centroids, thresholds = {}, {}
    for ci in np.unique(ytr):
        pts = Xtr[ytr == ci]
        c = pts.mean(axis=0)
        centroids[ci] = c
        thresholds[ci] = np.percentile(np.linalg.norm(pts - c, axis=1), 95)
    joblib.dump({'model': model, 'label_encoder': le, 'centroids': centroids,
                 'distance_thresholds': thresholds, 'feature_cols': feat_cols}, out_path)
    return acc, f1, ll, list(le.classes_)

def train_product():
    merged = pd.read_csv('merged_customer_dataset.csv')
    le_platform = LabelEncoder()
    merged['dominant_platform_code'] = le_platform.fit_transform(merged['dominant_platform'])
    cols = ['purchase_amount', 'customer_rating', 'avg_engagement_score', 'avg_purchase_interest_score',
            'avg_sentiment_score', 'platform_diversity', 'num_social_interactions',
            'purchase_month', 'is_high_value_purchase', 'dominant_platform_code']
    le_prod = LabelEncoder()
    X = merged[cols].values
    y = le_prod.fit_transform(merged['product_category'])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    cands = {'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE),
             'LogisticRegression': LogisticRegression(max_iter=1000)}
    best, best_acc, best_name = None, -1, None
    for name, clf in cands.items():
        clf.fit(Xtr, ytr)
        a = accuracy_score(yte, clf.predict(Xte))
        if a > best_acc:
            best, best_acc, best_name = clf, a, name
    joblib.dump({'model': best, 'label_encoder': le_prod, 'platform_encoder': le_platform,
                 'feature_cols': cols, 'model_name': best_name},
                'models/product_recommendation_model.joblib')
    return best_name, best_acc


if __name__ == '__main__':
    print('Regenerating image_features.csv from images/ ...')
    img_df = build_image_features()
    img_df.to_csv('image_features.csv', index=False)
    print('  ', img_df.shape, '| members:', sorted(img_df['member'].unique()))

    print('Regenerating audio_features.csv from audio/ ...')
    aud_df = build_audio_features()
    aud_df.to_csv('audio_features.csv', index=False)
    print('  ', aud_df.shape, '| members:', sorted(aud_df['member'].unique()))

    print('Training facial recognition model ...')
    a, f, l, cl = train_biometric(img_df, ['member', 'expression', 'augmentation', 'source_file', 'width', 'height'],
                                  'models/face_recognition_model.joblib', test_size=0.3)
    print(f'   face  -> acc={a:.3f} f1={f:.3f} logloss={l:.3f} classes={cl}')

    print('Training voiceprint model ...')
    a, f, l, cl = train_biometric(aud_df, ['member', 'phrase', 'augmentation', 'source_file'],
                                  'models/voice_verification_model.joblib', test_size=0.3)
    print(f'   voice -> acc={a:.3f} f1={f:.3f} logloss={l:.3f} classes={cl}')

    print('Training product recommendation model ...')
    name, acc = train_product()
    print(f'   product -> best={name} acc={acc:.3f}')
    print('Done. Models saved under models/.')
