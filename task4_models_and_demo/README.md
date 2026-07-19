# Task 4 & 5 — Models + System Demonstration

## Contents
- `task4_models.ipynb` — trains and evaluates all three models, saves them to `models/`
- `feature_extraction.py` — shared image/audio feature extraction (used identically at training time and inference time, so a live photo/recording is processed exactly like the training data)
- `cli_app.py` — the command-line app: face auth -> voice auth -> product recommendation
- `run_demo.sh` / `demo_transcript.txt` — the three required simulations (1 valid transaction, 2 unauthorized attempts) and their captured output
- `models/*.joblib` — the trained models + label encoders + open-set rejection thresholds
- `image_features.csv`, `audio_features.csv` — the four team identities plus an `UNAUTHORIZED_ATTEMPT` probe
- `merged_customer_dataset.csv` — copied from Task 1, used by the product recommendation model

## Identities in this folder

Facial recognition and voiceprint verification are only meaningful with **more than one identity** plus a genuine "unknown" test case. This folder therefore contains all four team members (`musana`, `amaliza`, `vestine`, `emmanuel`) plus:
- `UNAUTHORIZED_ATTEMPT` — a probe sample held out of training entirely, used only to test that the models correctly reject a face/voice they were never trained on.

**These are still synthetic placeholders.** When your group adds real data, keep one real folder per teammate under `images/` and `audio/` here (same `neutral.jpg`/`smiling.jpg`/`surprised.jpg` and `phrase1.wav`/`phrase2.wav` convention as Tasks 2–3), plus one genuine "unauthorized attempt" sample. Re-run `task4_models.ipynb` top to bottom — no code changes needed — then re-run `run_demo.sh`.

## How the open-set rejection works

A standard classifier always outputs *some* predicted class, even for a face/voice it's never seen. Both biometric models combine two checks before accepting an identity:
1. **Confidence threshold** on the classifier's own predicted probability
2. **Distance to that identity's training centroid**, thresholded at the 95th percentile of that identity's own training-sample spread

A sample must pass **both** to be accepted; otherwise the CLI prints `ACCESS DENIED`.

## Running the demo yourself

```bash
python3 cli_app.py --face images/musana/smiling.jpg --voice audio/musana/phrase2.wav
```

or re-run the whole scripted demo:

```bash
bash run_demo.sh
```

## Known limitations (worth stating plainly in your report)

- Facial/voiceprint accuracy is measured on **synthetic placeholder data** — real photos/recordings will behave differently (likely harder, more realistic).
- The product recommendation model's accuracy reflects genuinely weak correlation between the available social/transaction features and purchased category in this dataset (see Task 1's correlation heatmap) — not a bug.
- The face/voice biometrics identify your team, while the product step pulls from `merged_customer_dataset.csv` by `customer_id` (e-commerce customers) — a different identity space. State this simplification in your report.
- Face and voice identity are linked by the **enrolment label** (folder name), not a joint biometric model: two independent classifiers must independently output the same name.
