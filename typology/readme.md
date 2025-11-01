# Code for: Artificial Intelligence in Zooarchaeology: a Convolutional Neural Network to classify duck bones (bone typology classification)


## Associated Manuscript

This repository contains the full source code and configuration files to reproduce the results presented in the manuscript:

**Title:** Artificial Intelligence in Zooarchaeology: a Convolutional Neural Network to classify duck bones  
**Authors:** Elisa Paperini, elisa.paperini@phd.unipi.it; Beatrice Demarchi, beatrice.demarchi@unito.it; Nevio Dubbini, nevio.dubbini@unipi.it; Gabriele Gattiglia, gabriele.gattiglia@unipi.it; Lisa Yeomans, yeomans@sund.ku.dk; Claudia Sciuto, claudia.sciuto@unipi.it
  
**Journal:** Journal of Archaeological Science

---

## Description

This project implements a deep learning pipeline for the classification of avian skeletal elements by bone type. The workflow uses a pre-trained **VGG16** model, fine-tuned for this specific classification task. The process includes the following steps.

1.  **Data Preparation:** loading and augmenting the image dataset.
2.  **Hyperparameter Search:** a grid search to find a good configuration for learning rate, batch size, and other parameters.
3.  **Model Training:** training the customized VGG16 model on the image dataset.
4.  **Evaluation:** generating classification reports (including precision, recall, F1-score) and accuracy metrics for the training, validation, and test sets.

The code is contained within a single Python script for straightforward execution.

---

## File Descriptions

*   `TYPOLOGY_avifauna_classification.py` is the main executable script. It handles data loading, model definition (VGG16), the hyperparameter search, the final model training loop, evaluation, and saving the best model weights.
*   `requirements.txt` is a list of all Python packages and their specific versions required to run the code.

---

## System Requirements

*   **Python:** Version 3.10.4
*   **Key Python Packages:** See the requirements.txt file
*   **Hardware:** A CUDA-enabled GPU is highly recommended for reasonable training times. The code will automatically fall back to CPU if a GPU is not available.

---

## Installation and Usage

Follow these steps to set up the environment and run the code.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/elisa-paperini/AvifaunaBonesClassification
    cd AvifaunaBonesClassification/typology
    ```

2.  **Set up a Python Virtual Environment:**
    It is highly recommended to use a virtual environment to avoid conflicts with other projects.

    *Using `conda` (recommended):*
    ```bash
    conda create -n avifauna_repro python=3.10.4
    conda activate avifauna_repro
    ```

3.  **Install Dependencies:**
    Install all the required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare the Dataset:**
    This code expects the dataset to be organized in a specific folder structure, as used by `torchvision.datasets.ImageFolder`. See data in https://doi.org/10.5281/zenodo.17463327

5.  **Prepare the Data Directory:**
    Place your dataset folders in the `data` directory at the project root. The code expects the dataset to be organized in a folder structure compatible with `torchvision.datasets.ImageFolder`. See data in https://doi.org/10.5281/zenodo.17463327

6.  **Run the Main Script:**
    Execute the `TYPOLOGY_avifauna_classification.py` script to start the hyperparameter search and the final training process.
    ```bash
    python TYPOLOGY_avifauna_classification.py
    ```
    The script will print progress, epoch-wise results, and final classification reports to the console.