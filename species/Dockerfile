# Build from project root: docker build -f species/Dockerfile -t species-classification .

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

COPY species/requirements.txt /tmp/species_requirements.txt

RUN pip install --no-cache-dir torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

RUN pip install --no-cache-dir \
    optuna==4.5.0 \
    scikit-learn==1.7.1 \
    numpy==1.26.4 \
    pillow==11.3.0 \
    opencv-python==4.8.1.78 \
    pandas==2.3.1 \
    matplotlib==3.10.3 \
    seaborn==0.13.2 \
    scipy==1.15.3 \
    tqdm==4.67.1 \
    regex==2024.11.6 \
    pyunpack==0.3 \
    torch-optimizer==0.3.0 \
    torchsummary==1.5.1 \
    xgboost==3.0.5

COPY species/ /workspace/species/

RUN mkdir -p /workspace/data

WORKDIR /workspace/species

CMD ["python", "main.py"]

