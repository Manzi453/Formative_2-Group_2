#!/usr/bin/env python3
"""
Multimodal authentication + product recommendation CLI.

Flow (per the assignment spec):
  1. Face image  -> facial recognition model    -> reject immediately if unrecognized
  2. Voice sample -> voiceprint verification model -> reject immediately if unrecognized
  3. Both pass   -> run the product recommendation model and display the result

Usage:
  python cli_app.py --face path/to/face.jpg --voice path/to/voice.wav [--customer-id 178]
"""
import argparse
import sys
import numpy as np
import pandas as pd
import joblib

from feature_extraction import extract_image_feature_vector, extract_audio_feature_vector

FACE_MODEL_PATH = 'models/face_recognition_model.joblib'
VOICE_MODEL_PATH = 'models/voice_verification_model.joblib'
PRODUCT_MODEL_PATH = 'models/product_recommendation_model.joblib'
MERGED_DATA_PATH = 'merged_customer_dataset.csv'

PROBA_THRESHOLD = 0.5


def authenticate(feature_vector, bundle, proba_threshold=PROBA_THRESHOLD):
    model, le = bundle['model'], bundle['label_encoder']
    centroids, distance_thresholds = bundle['centroids'], bundle['distance_thresholds']

    feature_vector = np.asarray(feature_vector).reshape(1, -1)
    proba = model.predict_proba(feature_vector)[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    dist = float(np.linalg.norm(feature_vector[0] - centroids[pred_idx]))
    within_distance = dist <= distance_thresholds[pred_idx]

    if confidence >= proba_threshold and within_distance:
        return {'authorized': True, 'identity': le.inverse_transform([pred_idx])[0],
                'confidence': confidence, 'distance': dist}
    return {'authorized': False, 'identity': None, 'confidence': confidence, 'distance': dist}


def run_product_recommendation(customer_id=None):
    bundle = joblib.load(PRODUCT_MODEL_PATH)
    model, le_product, feature_cols = bundle['model'], bundle['label_encoder'], bundle['feature_cols']
    merged = pd.read_csv(MERGED_DATA_PATH)

    if customer_id is not None:
        row = merged[merged['customer_id'] == customer_id]
        if row.empty:
            print(f"  (no record for customer_id={customer_id}, sampling one instead)")
            row = merged.sample(1, random_state=None)
    else:
        row = merged.sample(1, random_state=None)

    x = row[feature_cols].values
    pred_idx = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    predicted_category = le_product.inverse_transform([pred_idx])[0]

    print(f"  Customer ID used for input features: {row['customer_id'].iloc[0]}")
    print(f"  Predicted product category: {predicted_category}  (confidence: {proba.max():.2f})")
    return predicted_category


def main():
    parser = argparse.ArgumentParser(description="Face + voice authentication -> product recommendation")
    parser.add_argument('--face', required=True, help='Path to a face image')
    parser.add_argument('--voice', required=True, help='Path to a voice recording')
    parser.add_argument('--customer-id', type=int, default=None, help='Optional customer_id to look up in merged dataset')
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: Facial recognition")
    face_bundle = joblib.load(FACE_MODEL_PATH)
    face_vec = extract_image_feature_vector(args.face, face_bundle['feature_cols'])
    face_result = authenticate(face_vec, face_bundle)
    print(f"  Input: {args.face}")
    print(f"  Result: {face_result}")

    if not face_result['authorized']:
        print("\n  ACCESS DENIED — face not recognized.")
        sys.exit(1)
    print(f"  Face recognized as: {face_result['identity']}")

    print("\nSTEP 2: Voiceprint verification")
    voice_bundle = joblib.load(VOICE_MODEL_PATH)
    voice_vec = extract_audio_feature_vector(args.voice, voice_bundle['feature_cols'])
    voice_result = authenticate(voice_vec, voice_bundle)
    print(f"  Input: {args.voice}")
    print(f"  Result: {voice_result}")

    if not voice_result['authorized']:
        print("\n  ACCESS DENIED — voice not recognized.")
        sys.exit(1)
    print(f"  Voice recognized as: {voice_result['identity']}")

    if face_result['identity'] != voice_result['identity']:
        print(f"\n  ACCESS DENIED — face identity ({face_result['identity']}) and "
              f"voice identity ({voice_result['identity']}) do not match.")
        sys.exit(1)

    print(f"\n  ACCESS GRANTED for {face_result['identity']}.")
    print("\nSTEP 3: Product recommendation")
    run_product_recommendation(args.customer_id)
    print("=" * 60)


if __name__ == '__main__':
    main()
