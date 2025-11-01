# Code for: Artificial Intelligence in Zooarchaeology: a Convolutional Neural Network to classify duck bones (species classification)


## Associated Manuscript

This repository contains the full source code and configuration files to reproduce the results presented in the manuscript:

**Title:** Artificial Intelligence in Zooarchaeology: a Convolutional Neural Network to classify duck bones  
**Authors:** Elisa Paperini, elisa.paperini@phd.unipi.it; Beatrice Demarchi, beatrice.demarchi@unito.it; Nevio Dubbini, nevio.dubbini@unipi.it; Gabriele Gattiglia, gabriele.gattiglia@unipi.it; Lisa Yeomans, yeomans@sund.ku.dk; Claudia Sciuto, claudia.sciuto@unipi.it
  
**Journal:** Journal of Archaeological Science

---

## Description

This project implements a deep learning pipeline for the taxonomic classification of avian skeletal elements. The workflow uses a pre-trained ResNet-101 model, fine-tuned for the specific classification task. The process includes:

1.  **Hyperparameter Optimization:** Using Optuna to find the optimal learning rate, batch size, and momentum.
2.  **Model Training:** Training the customized ResNet-101 model on the image dataset.
3.  **Evaluation:** Generating classification reports (including precision, recall, F1-score) and accuracy metrics (Top-1, Top-3, Top-5) for both training and validation sets for each epoch.

The code is structured to be modular, with separate files for the main script, model architecture, and utility functions.

---

## File Descriptions

*   `main.py` is the main executable script. It handles data loading, hyperparameter search with Optuna, the final model training loop, and saving the best model weights.
*   `models.py` defines the custom classification head that replaces the final layer of the ResNet model and the weight initialization function.
*   `utils.py` contains helper functions, including the objective function for Optuna trials and the core `train_and_eval_cnn` logic.
*   `data.py` contains the `N_AugmentedDataset` class, a wrapper to apply data augmentation multiple times to each image.
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
    cd AvifaunaBonesClassification/species
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
    Execute the `main.py` script to start the hyperparameter search and the final training process.
    ```bash
    python main.py
    ```
    The script will print progress, epoch-wise results, and final classification reports to the console.