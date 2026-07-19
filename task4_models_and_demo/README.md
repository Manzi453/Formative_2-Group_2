# Task 4 & 5 — Models + System Demonstration

## Contents
- `task4_models.ipynb` — trains and evaluates all three models, saves them to `models/`
- `feature_extraction.py` — shared image/audio feature extraction (used identically at training time and inference time, so a live photo/recording is processed exactly like the training data)
- `cli_app.py` — the command-line app: face auth -> voice auth -> product recommendation
- `run_demo.sh` / `demo_transcript.txt` — the three required simulations (1 valid transaction, 2 unauthorized attempts) and their captured output
- `models/*.joblib` — the trained models + label encoders + open-set rejection thresholds
- `image_features.csv`, `audio_features.csv` — the **real team features** (4 members) from Tasks 2–3 + one held-out `UNAUTHORIZED_ATTEMPT` probe (see note below)
- `merged_customer_dataset.csv` — copied from Task 1, used by the product recommendation model

## Identities in this folder

The facial-recognition and voiceprint models here are trained on the **real team data** collected in Tasks 2–3 — one authorized identity per member:
- `amaliza`, `emmanuel`, `musana`, `vestine` — each with 3 facial expressions (`neutral`/`smiling`/`surprised`) and 2 spoken phrases (`phrase1.wav` = "Yes, approve", `phrase2.wav` = "Confirm transaction"), expanded by the Task 2/3 augmentations.
- `UNAUTHORIZED_ATTEMPT` — a single out-of-group probe sample, **held out of training entirely**, used only to verify the models reject an unknown identity. This is the one remaining placeholder: swap it for a real photo/recording of someone not on the team (or an unclear/obstructed shot) before the final submission.

The `images/` and `audio/` files here are copied from Tasks 2–3 (vestine's `.jpeg` photos and musana's differently-named recordings are normalized to the shared `neutral.jpg`/`phrase1.wav` convention so training paths, the feature CSVs, and the CLI all agree). Re-running `task4_models.ipynb` top to bottom retrains on whatever member folders exist — no code changes needed — then re-run `run_demo.sh`.

## How the open-set rejection works

A standard classifier always outputs *some* predicted class, even for a face/voice it's never seen. Both biometric models combine two checks before accepting an identity:
1. **Confidence threshold** on the classifier's own predicted probability
2. **Distance to that identity's training centroid**, thresholded at the 95th percentile of that identity's own training-sample spread

A sample must pass **both** to be accepted; otherwise the CLI prints `ACCESS DENIED`.

## Running the demo yourself

```bash
python3 cli_app.py --face images/emmanuel/smiling.jpg --voice audio/emmanuel/phrase2.wav
```

or re-run the whole scripted demo:

```bash
bash run_demo.sh
```

## Known limitations (worth stating plainly in your report)

- Facial/voiceprint accuracy is measured on the **real team data** (4 members), but each member contributes only a few source samples expanded by augmentation. The stratified train/test split can place near-duplicate augmented variants on both sides, so the high biometric scores show the pipeline cleanly separates *these* members — not a large-scale benchmark. More sessions/photos per member would give a firmer estimate.
- The `UNAUTHORIZED_ATTEMPT` probe is currently a single synthetic out-of-group sample. Replace it with a genuine non-member photo/recording for the strongest open-set demonstration.
- The product recommendation model's modest accuracy reflects genuinely weak correlation between the available social/transaction features and purchased category in this dataset (see Task 1's correlation heatmap) — not a bug. More/better features or more transaction history per customer would be needed for real predictive power.
- The CLI's product-recommendation step pulls from `merged_customer_dataset.csv` by `customer_id`, which is **not** the same identity space as the face/voice biometrics (one is e-commerce customers, the other is your team) — a simplification made because the assignment's two data sources (customer transactions and team biometrics) aren't naturally linked. State this explicitly as a design decision in your report.
