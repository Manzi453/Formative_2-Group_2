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
    ├── models/*.joblib               # trained models
    ├── image_features.csv / audio_features.csv   # 4 members + 1 UNAUTHORIZED_ATTEMPT probe
    ├── images/ , audio/              # PLACEHOLDERS — replace with real team data
    └── README.md                     # design notes specific to Tasks 4–5
```

Team members: **Musana, Amaliza, Vestine, Emmanuel**.

## IMPORTANT — what's real vs placeholder

- **Real data:** `customer_transactions.xlsx` / `customer_social_profiles.xlsx` (provided by the course) and everything derived from them — `merged_customer_dataset.csv` and the product recommendation model are trained on genuine data.
- **Placeholder data:** every face image and audio recording in this repo is **synthetically generated** (simple drawings / generated tones). Each team member (`musana`, `amaliza`, `vestine`, `emmanuel`) has a folder, but the files inside are placeholders that only prove the pipeline runs end-to-end. **Your group must replace `images/<name>/` and `audio/<name>/` in Tasks 2, 3, and 4 with each member's real photos and recordings before submitting** — that is the actual deliverable the rubric grades (Image/Audio Quantity & Diversity, Augmentation & Feature Extraction).

Once real data is in place, every notebook re-runs top-to-bottom with **no code changes** — the pipelines auto-discover whatever member folders exist.

## Suggested order of work for your group

1. Each member takes 3 face photos (neutral/smiling/surprised) + 2 voice recordings (the two required phrases), saved with the exact filenames `neutral.jpg`/`smiling.jpg`/`surprised.jpg` and `phrase1.wav`/`phrase2.wav` inside their own `images/<name>/` and `audio/<name>/` folder — **using the same folder name in both places** so the face and voice models share one identity label per person.
2. Also capture one genuine "unauthorized attempt" sample (someone outside the group, or an unclear/obstructed shot) for the open-set rejection test — keep it in the `UNAUTHORIZED_ATTEMPT/` folder.
3. Drop everything into `task2_images/images/`, `task3_audio/audio/`, and `task4_models_and_demo/images/` + `audio/`.
4. Re-run all four notebooks top-to-bottom in order.
5. Re-run `task4_models_and_demo/run_demo.sh` for your system-demonstration video.
6. Finish the report (see `Formative2_Report.docx`) — fill in each member's contribution, update the biometric results after re-running on real data, insert the figures, and export to PDF.

## What's still on your group to produce

- The actual real photos/recordings (see above)
- The written report's team-contribution split, and the biometric results after real data is added
- The system-demonstration video (record yourselves running `run_demo.sh` with real data)
- The GitHub repository itself + each member's contribution summary
