#!/bin/bash
# System demonstration script for Task 5.
# Run from inside this folder: bash run_demo.sh

echo "########## DEMO 1: VALID / AUTHORIZED TRANSACTION (emmanuel) ##########"
python3 cli_app.py --face images/emmanuel/smiling.png --voice audio/emmanuel/phrase2.wav --customer-id 178
echo
echo "########## DEMO 2: UNAUTHORIZED ATTEMPT -- unrecognized face ##########"
python3 cli_app.py --face images/UNAUTHORIZED_ATTEMPT/neutral.jpg --voice audio/emmanuel/phrase1.wav
echo
echo "########## DEMO 3: UNAUTHORIZED ATTEMPT -- unrecognized voice (face alone is not enough) ##########"
python3 cli_app.py --face images/amaliza/neutral.jpg --voice audio/UNAUTHORIZED_ATTEMPT/phrase1.wav
