# Dockerfile — reproducible environment for the Multimodal Data Preprocessing pipeline.
# Pins Python and every library to the exact, mutually-compatible versions the
# project was tested on, so `pip` never resolves a conflicting combination
# (the usual offender is numpy <-> numba/llvmlite, which is locked here).
#
# Build:   docker build -t formative2 .
# Shell:   docker run --rm -it -v "$PWD":/app formative2
# Jupyter: docker run --rm -it -p 8888:8888 -v "$PWD":/app formative2 \
#            jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# Demo:    docker run --rm -it -v "$PWD":/app -w /app/task4_models_and_demo formative2 bash run_demo.sh

FROM python:3.12-slim

# OS libraries needed at runtime:
#   libsndfile1 -> soundfile / librosa (reading .wav)
#   libgomp1    -> OpenMP runtime used by scikit-learn
#   ffmpeg      -> librosa / audioread fallback for non-wav audio
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Exact, tested, conflict-free dependency set.
RUN python -m pip install --upgrade pip && \
    pip install \
        numpy==2.4.4 \
        scipy==1.17.1 \
        llvmlite==0.48.0 \
        numba==0.66.0 \
        pandas==3.0.2 \
        scikit-learn==1.8.0 \
        scikit-image==0.26.0 \
        matplotlib==3.10.8 \
        seaborn==0.13.2 \
        pillow==12.1.1 \
        soundfile==0.14.0 \
        audioread==3.1.0 \
        librosa==0.11.0 \
        joblib==1.5.3 \
        openpyxl==3.1.5 \
        nbformat==5.10.4 \
        nbconvert==7.17.1 \
        ipykernel==7.3.0

# Copy the whole project (notebooks, scripts, data, models) into the image.
COPY . /app

CMD ["bash"]