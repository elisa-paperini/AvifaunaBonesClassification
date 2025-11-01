# Docker Usage Guide

This project includes Dockerfiles for both the species and typology classification projects.

## Prerequisites

- Docker installed and running
- NVIDIA Docker runtime (nvidia-docker2) for GPU support
- CUDA-capable GPU (recommended but not required, code will fall back to CPU)

## Building the Docker Images

Build from the project root directory:

### Species Classification
```bash
docker build -f species/Dockerfile -t species-classification .
```

### Typology Classification
```bash
docker build -f typology/Dockerfile -t typology-classification .
```

## Running the Containers

### Species Classification

Mount your data directory and run:
```bash
docker run --gpus all -v /path/to/your/data:/workspace/data species-classification
```

Or if you don't have GPU support:
```bash
docker run -v /path/to/your/data:/workspace/data species-classification
```

### Typology Classification

Mount your data directory and run:
```bash
docker run --gpus all -v /path/to/your/data:/workspace/data typology-classification
```

Or if you don't have GPU support:
```bash
docker run -v /path/to/your/data:/workspace/data typology-classification
```

## Data Directory Structure

Place your datasets in the mounted data directory with the following structure:

For species classification:
```
data/
└── bones_detection_species_BN
```

For typology classification:
```
data/
└── bones_detection_typology
```

## Saving Model Outputs

To save model checkpoints and outputs, mount an additional volume:

```bash
docker run --gpus all \
  -v /path/to/your/data:/workspace/data \
  -v /path/to/output:/workspace/output \
  species-classification
```

The models will be saved in the working directory (`/workspace/species` or `/workspace/typology`), so you can access them via the mounted output volume.

## Interactive Mode

To run interactively for debugging:
```bash
docker run --gpus all -it --entrypoint /bin/bash \
  -v /path/to/your/data:/workspace/data \
  species-classification
```

