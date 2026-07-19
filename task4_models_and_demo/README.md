# Task 4 & 5 — Models + System Demonstration

## Contents
- `task4_models.ipynb` — trains and evaluates all three models, saves them to `models/`
- `feature_extraction.py` — shared image/audio feature extraction (used identically at training time and inference time, so a live photo/recording is processed exactly like the training data)
- `cli_app.py` — the command-line app: face auth -> voice auth -> product recommendation
- `run_demo.sh` / `demo_transcript.txt` — the three required simulations (1 valid transaction, 2 unauthorized attempts) and their captured output
- `models/*.joblib` — the trained models + label encoders + open-set rejection thresholds
- `image_features.csv`, `audio_features.csv` — extended with a second placeholder identity + an `UNAUTHORIZED_ATTEMPT` probe (see note below)
- `merged_customer_dataset.csv` — copied from Task 1, used by the product recommendation model

## Why this folder has extra placeholder identities beyond Task 2/3

Facial recognition and voiceprint verification are only meaningful with **more than one class** to discriminate between, plus a genuine "unknown" test case — a single identity gives trivially perfect accuracy and nothing to reject. So this folder's `images/` and `audio/` extend Task 2/3's placeholder set with:
- `EXAMPLE_MEMBER_2` — a second synthetic identity (still not a real photo/recording)
- `UNAUTHORIZED_ATTEMPT` — a synthetic probe sample, held out of training entirely, used only to test that the models correctly reject it

**When your group adds real data:** put one real folder per teammate under `images/` and `audio/` here (same `neutral.jpg`/`smiling.jpg`/`surprised.jpg` and `phrase1.wav`/`phrase2.wav` convention as Tasks 2–3), plus one genuine "unauthorized attempt" sample (a photo/recording of someone *not* on the team, or an unclear/obstructed shot). Re-run `task4_models.ipynb` top to bottom — no code changes needed — then re-run `run_demo.sh` pointing at real file paths.

## How the open-set rejection works

A standard classifier always outputs *some* predicted class, even for a face/voice it's never seen. Both biometric models combine two checks before accepting an identity:
1. **Confidence threshold** on the classifier's own predicted probability
2. **Distance to that identity's training centroid**, thresholded at the 95th percentile of that identity's own training-sample spread

A sample must pass **both** to be accepted; otherwise the CLI prints `ACCESS DENIED`.

## Running the demo yourself

```bash
python3 cli_app.py --face images/EXAMPLE_MEMBER/smiling.jpg --voice audio/EXAMPLE_MEMBER/phrase2.wav
```

or re-run the whole scripted demo:

```bash
bash run_demo.sh
```

## Known limitations (worth stating plainly in your report)

- Facial/voiceprint accuracy is measured on **synthetic placeholder data** — real photos/recordings will behave differently (likely harder, more realistic).
- The product recommendation model's accuracy (~17%, roughly chance for 5 classes) reflects genuinely weak correlation between the available social/transaction features and purchased category in this dataset (see Task 1's correlation heatmap) — not a bug. More/better features or more transaction history per customer would be needed for real predictive power.
- The CLI's product-recommendation step pulls from `merged_customer_dataset.csv` by `customer_id`, which is **not** the same identity space as the face/voice biometrics (one is e-commerce customers, the other is your team) — a simplification made because the assignment's two data sources (customer transactions and team biometrics) aren't naturally linked. State this explicitly as a design decision in your report.
