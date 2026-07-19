# Formative 2 — Multimodal Data Preprocessing

## Repository structure

```
formative2_package/
├── requirements.txt
├── task1_data_merge/
│   ├── task1_data_merge.ipynb        # EDA, cleaning, merge, validation, feature engineering
│   └── merged_customer_dataset.csv   # real data — output of Task 1
├── task2_images/
│   ├── task2_image_pipeline.ipynb    # load, display, augment, extract image features
│   ├── image_features.csv
│   └── images/{musana,amaliza,vestine,emmanuel}/   # PLACEHOLDERS — replace with real team photos
├── task3_audio/
│   ├── task3_audio_pipeline.ipynb    # load, visualize, augment, extract audio features
│   ├── audio_features.csv
│   └── audio/{musana,amaliza,vestine,emmanuel}/    # PLACEHOLDERS — replace with real team recordings
└── task4_models_and_demo/
    ├── task4_models.ipynb            # trains & evaluates all 3 models
    ├── feature_extraction.py         # shared feature logic (training + live inference)
    ├── cli_app.py                    # the command-line authentication + recommendation app
    ├── run_demo.sh / demo_transcript.txt   # the 3 required system-demo simulations
    ├── models/*.joblib               # trained models (real 4 members)
    ├── image_features.csv / audio_features.csv   # 4 real members + 1 UNAUTHORIZED_ATTEMPT probe
    ├── images/ , audio/              # real team photos/recordings (from Tasks 2–3) + 1 unauthorized probe
    └── README.md                     # design notes specific to Tasks 4–5
```

Team members: Ivan Manzi Musana
              ,Amaliza shalom 
              ,Vestine Umukundwa 
              ,Emmanuel Nsabagasani.

## IMPORTANT — what's real vs placeholder

### Model Performance Summary
- **Face Recognition Model:** Trained on 3 expressions × 4 members with augmentation
- **Voice Verification Model:** Trained on 2 phrases × 4 members with augmentation  
- **Product Recommendation Model:** Trained on customer transaction data
- All models implemented with open-set rejection to detect unauthorized users

- **Real data:** `customer_transactions.xlsx` / `customer_social_profiles.xlsx` (provided by the course) and everything derived from them — `merged_customer_dataset.csv` and the product recommendation model are trained on genuine data. The **face images and audio recordings for all four members** (`amaliza`, `emmanuel`, `musana`, `vestine`) in Tasks 2, 3, and 4 are the team's real photos and recordings, and the facial-recognition / voiceprint models in Task 4 are trained on them.
- **One remaining placeholder:** the single `UNAUTHORIZED_ATTEMPT/` probe under `task4_models_and_demo/` is a synthetic out-of-group sample used only to test open-set rejection. Swap it for a genuine photo/recording of someone *not* on the team (or an unclear/obstructed shot) for the strongest unauthorized-attempt demo.

Every notebook re-runs top-to-bottom with **no code changes** — the pipelines auto-discover whatever member folders exist.

#### Security Testing Strategy
- Open-set rejection verified using UNAUTHORIZED_ATTEMPT probes
- Both confidence threshold and centroid distance checks required
- Multi-modal biometric fusion (face + voice) for transaction authorization

## Status of work

Done and committed:

1. All four members' real face photos (neutral/smiling/surprised) and voice recordings (phrase1 = "Yes, approve", phrase2 = "Confirm transaction") are in `task2_images/`, `task3_audio/`, and `task4_models_and_demo/` under one folder per member.
2. Tasks 1–4 notebooks re-run top-to-bottom on the real data; the three Task 4 models are trained on it and saved under `task4_models_and_demo/models/`.
3. `task4_models_and_demo/run_demo.sh` runs the full valid-transaction + two unauthorized-attempt demos on real members; `demo_transcript.txt` holds the captured output.
4. **All biometric data validated** — facial recognition and voice verification models successfully trained on team member data.

## What's still on your group to produce

- Replace the single synthetic `UNAUTHORIZED_ATTEMPT/` probe with a genuine out-of-group photo/recording (optional but stronger for the demo), then re-run `task4_models.ipynb` + `run_demo.sh`.
- The written report's team-contribution split and the biometric-results write-up (metrics are already in the notebook outputs).
- The system-demonstration video (record yourselves running `run_demo.sh`).
- Push the repository to GitHub with each member's contribution summary.


### Environment note
The notebooks were executed with a local virtualenv (`.venv/`, `pip install -r requirements.txt`) registered as a Jupyter kernel. Any Python 3.10–3.13 environment with `requirements.txt` installed works.
