### Load libraries

# OS and file handling libraries
import os
import glob
import time
import pathlib
from PIL import Image
import regex as re

# Numerical computing and data manipulation
import numpy as np
import itertools # Useful for iterating over multiple variables

# PyTorch for deep learning
import torch
import torch.nn as nn # Neural network modules
import torch.nn.functional as F # Functional API for layers and activation functions
from torch.utils.data import Dataset # Data loading utilities
from torch.optim import Adam, lr_scheduler # Optimizer and learning rate scheduler

# Torchvision for image processing and pre-trained models
import torchvision
from torchvision import models # Datasets, pre-trained models, and data transformations
import torchvision.utils # Utility functions for visualization
import torchvision.datasets as dsets # Alternative dataset module
from torchvision.transforms import v2  # Advanced image transformations (newer API in torchvision)

# Scikit-learn for model evaluation metrics and k-fold cross-validation
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

import shutil

# Library for extracting .zip archives
from pyunpack import Archive  # Used for extracting compressed files




def main():

    #----- DEVICE SETUP -----#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print('------------------------')
    print('Selected device:', device)
    print('------------------------')
    print(' ')
    since = time.time()


    #----- DATA PATHS -----#
    # Define the data directories (*change with your actual path*)
    data_dir = 'F:/elisa/dataset_ossa/bones_detection_typology'
    training_dir = os.path.join(data_dir, 'bones_train')
    validation_dir =  os.path.join(data_dir, 'bones_validation')

    #----- DATA PRE-PROCESSING & DATASETS -----#
    # Preprocessing steps applied to training data
    train_transform = v2.Compose([
        v2.Resize(size=(224, 224), antialias=True), # Resize images, using antialiasing
        v2.RandomRotation(degrees=(-2, 2)), # Randomly rotate images in the [-2. 2] degrees interval
        v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5)), # Apply a Gaussian blur to the images
        v2.ToTensor(),  # Change the pixel range from 0-255 to 0-1, numpy to tensors
        v2.ToDtype(torch.float32),#, scale=True), # Convert the tensor to the torch.float32 data type required for PyTorch
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the image tensor using the specified mean and standard deviation
        ])

    # Preprocessing steps applied to validation data
    val_transform = v2.Compose([
        v2.Resize(size=(224, 224), antialias=True),  # Resize images, using antialiasing
        v2.ToTensor(),  # Change the pixel range from 0-255 to 0-1, numpy to tensors
        v2.ToDtype(torch.float32),# scale=True),    # Convert the tensor to the torch.float32 data type required for PyTorch
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the image tensor using the specified mean and standard deviation
        ])

    # Load original images and labels, apply transforms for training and validation
    train_data = dsets.ImageFolder(root=training_dir, transform=train_transform)
    val_data = dsets.ImageFolder(root=validation_dir, transform=val_transform)

    # Loading dataset for training and validation
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2, persistent_workers=True) # Create the training data loader
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=True, num_workers=2, persistent_workers=True) # Create the validation data loader

    # Print information about the training dataset
    print('Training Dataset:')
    print(f'Total samples: {len(train_loader.dataset)}')  # Display the number of samples in the training set
    print(f'Dataset type: {type(train_loader.dataset)}')  # Show the dataset type

    # Print information about the validation dataset
    print('Validation Dataset:')
    print(f'Total samples: {len(val_loader.dataset)}')  # Display the number of samples in the validation set
    print(f'Dataset type: {type(val_loader.dataset)}')  # Show the dataset type
    print('------')

    # Calculate the number of training and validation images
    train_count=len(glob.glob(os.path.join(training_dir, '**', '*.jpg'), recursive=True)) # Count all JPG files recursively in the training directory
    val_count=len(glob.glob(os.path.join(validation_dir, '**', '*.jpg'), recursive=True)) # Count all JPG files recursively in the validation directory

    print('Number of images in train dataset:', train_count)
    print('Number of images in validation dataset:', val_count)

    # Retrieve the class categories (labels)
    # Get classes directly from the dataset object for robustness
    classes = train_data.classes

    print('Class labels:', classes)
    print('Total number of classes:', len(classes))
    num_classes = len(classes)


    # Define a grid of hyperparameters
    learning_rates = [0.001, 0.0001] # Different learning rates to test
    batch_sizes = [8, 16] # Different batch sizes to test
    weight_decay= [0.001, 0.0001] # Different weight decay values to test
    step_size= [3, 10] # Step size values for learning rate scheduler

    # Initialize variables to track the best model configuration
    best_accuracy = 0.0 # Stores the highest validation accuracy achieved
    best_hyperparams = {} # Dictionary to store the best hyperparameters
    num_epochs = 20 # Set the number of epochs for training

    # Store results for each combination
    results = []

    # Iterate over all possible combinations of hyperparameters
    for lr, batch_size, weight_decay, step_size, in itertools.product(learning_rates, batch_sizes, weight_decay, step_size):
        print('\n----------------------------------\n')
        print(f"Training with lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, step_size={step_size}")
        print('\n----------------------------------\n')
    
        # --- FIX: Re-initialize model for each hyperparameter trial ---
        model_ft = models.vgg16(weights='IMAGENET1K_V1')
        for param in model_ft.parameters():
            param.requires_grad = False

        # Get the number of input features for the final classifier layer
        n_inputs = model_ft.classifier[6].in_features

        # --- FIX: Use dynamic number of classes ---
        # Replace the last classifier layer with a custom fully connected layer
        model_ft.classifier[6] = nn.Linear(in_features=n_inputs, out_features=num_classes)

        # Move the model to the specified device (CPU or GPU)
        model_ft = model_ft.to(device)

        # Create data loaders with the current batch size
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) # Training data loader
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False) # Validation data loader

        # Re-initialize the optimizer with the current learning rate
        optimizer_ft = Adam(model_ft.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        loss_function_ft = nn.CrossEntropyLoss() # Loss function for multi-class classification
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.1) # Learning rate scheduler
        # early_stopper = EarlyStopper(patience=5, min_delta=10) # Initialize early stopping mechanism

        # Training loop
        for epoch in range(num_epochs):
            model_ft.train() # Set model to training mode
            train_accuracy = 0.0
            train_loss = 0.0

            for images, labels in train_loader: # Iterate over training batches
                images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
                optimizer_ft.zero_grad() # Reset gradients
                outputs = model_ft(images) # Forward pass
                loss = loss_function_ft(outputs, labels) # Compute loss
                loss.backward() # Backpropagate
                optimizer_ft.step() # Update weights

                # --- FIX: Use loss.item() for safer loss accumulation ---
                train_loss += loss.item() * images.size(0) # Accumulate training loss
                _, prediction = torch.max(outputs.data, 1) # Get predicted class labels
                train_accuracy += int(torch.sum(prediction == labels.data)) # Count correct predictions

            train_accuracy /= len(train_data) # Compute training accuracy
            train_loss /= len(train_data) # Compute average training loss

            # Evaluate the model on validation set
            model_ft.eval() # Set model to evaluation mode
            val_accuracy = 0.0
            val_loss = 0.0

            with torch.no_grad(): # Disable gradient computation for validation
                for images, labels in val_loader: # Iterate over validation batches
                    images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
                    outputs = model_ft(images) # Forward pass
                    loss = loss_function_ft(outputs, labels) # Compute validation loss
                    val_loss += loss.item() * images.size(0) # Accumulate validation loss
                    _, prediction = torch.max(outputs.data, 1) # Get predicted class labels
                    val_accuracy += int(torch.sum(prediction == labels.data)) # Count correct predictions
            val_accuracy /= len(val_data) # Compute validation accuracy
            val_loss /= len(val_data) # Compute average validation loss

            # Print training and validation metrics
            print(f'Epoch: {epoch} Train Accuracy: {train_accuracy:.2f} Train Loss: {train_loss:.2f} Val. Accuracy: {val_accuracy:.2f} Val. Loss: {val_loss:.2f}')

            # # Check early stopping condition
            # if early_stopper.early_stop(val_loss):
            #     break # Stop training if validation loss does not improve

            # Save the model if it achieves the best validation accuracy so far
            if val_accuracy > best_accuracy:
                torch.save(model_ft.state_dict(), 'best_checkpoint_ft.model')  # Save the model state
                best_accuracy = val_accuracy # Update best accuracy
                best_hyperparams = {'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay, 'step_size': step_size} # Store best hyperparameters

        # Append results for the current combination of hyperparameters
        results.append((lr, batch_size, num_epochs, val_accuracy))

    # Print the best hyperparameter configuration found
    print('\n----------------------------------\n')
    print(f'Best hyperparameters: {best_hyperparams} with validation accuracy: {best_accuracy}')
    print('\n----------------------------------\n')





    #----- TRAINING WITH BEST HYPERPARAMETERS -----#

    # Initialization of lists to store training and validation metrics
    summary_loss_train = [] # Stores training loss for each epoch
    summary_acc_train = [] # Stores training accuracy for each epoch
    summary_loss_val = [] # Stores validation loss for each epoch
    summary_acc_val = [] # Stores validation accuracy for each epoch
    summary_precision_train = [] # Stores training precision for each epoch
    summary_recall_train = [] # Stores training recall for each epoch
    summary_f1_train = [] # Stores training F1-score for each epoch
    summary_precision_val = [] # Stores validation precision for each epoch
    summary_recall_val = [] # Stores validation recall for each epoch
    summary_f1_val = [] # Stores validation F1-score for each epoch

    # Load the pre-trained VGG16 model with ImageNet weights
    model_ft = models.vgg16(weights='IMAGENET1K_V1')

    # Freeze all layers of the pre-trained model
    for param in model_ft.parameters():
        param.requires_grad = False # Prevents updates to pre-trained weights

    # Modify the final fully connected layer for the classification task
    n_inputs = model_ft.classifier[6].in_features # Get the input size of the final layer
    # --- FIX: Use dynamic number of classes ---
    model_ft.classifier[6] = nn.Linear(in_features=n_inputs, out_features=num_classes) # Replace last layer
    model_ft = model_ft.to(device) # Move model to GPU if available

    # --- FIX: Use the best hyperparameters found during the search ---
    final_batch_size = best_hyperparams.get('batch_size', 8) # Use found batch size, or default to 8
    final_lr = best_hyperparams.get('lr', 0.001)
    final_weight_decay = best_hyperparams.get('weight_decay', 0.001)
    final_step_size = best_hyperparams.get('step_size', 10)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=final_batch_size, shuffle=True) # Training data loader
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=final_batch_size, shuffle=False) # Validation data loader

    optimizer_ft = Adam(model_ft.classifier.parameters(), lr=final_lr, weight_decay=final_weight_decay) # Adam optimizer
    loss_function_ft = nn.CrossEntropyLoss() # Cross-entropy loss function for classification
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=final_step_size, gamma=0.1) # Learning rate scheduler

    best_accuracy = 0.0 # Track the best validation accuracy
    num_epochs = 70 # Set a higher number of epochs for final training

    # Training loop for num_epochs epochs
    for epoch in range(num_epochs):
        model_ft.train() # Set model to training mode
        train_accuracy = 0.0
        train_loss = 0.0
        all_train_labels = []
        all_train_preds = []

        for images, labels in train_loader: # Iterate through training batches
            images, labels = images.to(device), labels.to(device) # Move data to GPU if available
            optimizer_ft.zero_grad() # Reset gradients
            outputs = model_ft(images) # Forward pass
            loss = loss_function_ft(outputs, labels) # Compute loss
            loss.backward() # Backpropagation
            optimizer_ft.step() # Update weights

            # --- FIX: Use loss.item() ---
            train_loss += loss.item() * images.size(0) # Accumulate loss
            _, prediction = torch.max(outputs.data, 1) # Get predicted class
            train_accuracy += int(torch.sum(prediction == labels.data)) # Count correct predictions
            all_train_labels.extend(labels.cpu().numpy()) # Store true labels
            all_train_preds.extend(prediction.cpu().numpy()) # Store predicted labels

        train_accuracy /= len(train_data) # Compute training accuracy
        train_loss /= len(train_data) # Compute average training loss

        # Compute precision, recall, and F1-score for training data
        train_precision = precision_score(all_train_labels, all_train_preds, average='macro')
        train_recall = recall_score(all_train_labels, all_train_preds, average='macro')
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')

        # Validation phase
        model_ft.eval() # Set model to evaluation mode
        val_accuracy = 0.0
        val_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad(): # Disable gradient computation for validation
            for images, labels in val_loader: # Iterate through validation batches
                images, labels = images.to(device), labels.to(device) # Move data to GPU if available
                outputs = model_ft(images) # Forward pass
                loss = loss_function_ft(outputs, labels) # Compute validation loss
                val_loss += loss.item() * images.size(0) # Accumulate loss
                _, prediction = torch.max(outputs.data, 1) # Get predicted class
                val_accuracy += int(torch.sum(prediction == labels.data)) # Count correct predictions
                all_val_labels.extend(labels.cpu().numpy()) # Store true labels
                all_val_preds.extend(prediction.cpu().numpy()) # Store predicted labels

        val_accuracy /= len(val_data) # Compute validation accuracy
        val_loss /= len(val_data) # Compute average validation loss

        # Compute precision, recall, and F1-score for validation data
        val_precision = precision_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average='macro')
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')

        # Store training and validation metrics
        summary_loss_train.append(train_loss)
        summary_acc_train.append(train_accuracy)
        summary_precision_train.append(train_precision)
        summary_recall_train.append(train_recall)
        summary_f1_train.append(train_f1)

        summary_loss_val.append(val_loss)
        summary_acc_val.append(val_accuracy)
        summary_precision_val.append(val_precision)
        summary_recall_val.append(val_recall)
        summary_f1_val.append(val_f1)

        # Print training and validation results for the current epoch
        print(f'Epoch: {epoch} Train Accuracy: {train_accuracy:.4f} Train Loss: {train_loss:.4f} '
            f'Val. Accuracy: {val_accuracy:.4f} Val. Loss: {val_loss:.4f} '
            f'Train Precision: {train_precision:.4f} Train Recall: {train_recall:.4f} Train F1: {train_f1:.4f} '
            f'Val Precision: {val_precision:.4f} Val Recall: {val_recall:.4f} Val F1: {val_f1:.4f}')

        # Save the model if it achieves the best validation accuracy so far
        if val_accuracy > best_accuracy:
            torch.save(model_ft.state_dict(), 'best_checkpoint_ft.model') # Save model state
            best_accuracy = val_accuracy # Update best validation accuracy

    # Print the highest validation accuracy achieved
    print('\n----------------------------------\n')    
    print(f'Best validation accuracy: {best_accuracy}')
    print('\n----------------------------------\n')





    #----- TESTING PHASE -----#

    # Path to test directory
    test_path =  os.path.join(data_dir, 'bones_test')
    test_data = dsets.ImageFolder(root=test_path, transform=val_transform)
    # --- FIX: Use final_batch_size and set shuffle=False for testing ---
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=final_batch_size, shuffle=False, num_workers=2)
    
    # Count the number of test images
    test_count=len(glob.glob(os.path.join(test_path, '**', '*.jpg'), recursive=True))
    print(test_count) # Print the total number of test images

    print("\n--- Loading best model for testing ---")
    # Initialize a VGG16 model
    model_ft = models.vgg16() # No need to download weights again
    # Modify the classifier to have the correct number of output classes
    n_inputs = model_ft.classifier[6].in_features
    # --- FIX: Use dynamic number of classes ---
    model_ft.classifier[6] = nn.Linear(n_inputs, num_classes)

    # Load the best model weights saved during training
    checkpoint = torch.load('best_checkpoint_ft.model')
    model_ft.load_state_dict(checkpoint)
    model_ft.to(device) # Move model to device

    # Calculate overall metrics for the test set
    model_ft.eval() # Set the model to evaluation mode (disables dropout and batch normalization)
    test_accuracy = 0.0
    test_loss = 0.0
    all_test_labels = []
    all_test_preds = []

    with torch.no_grad(): # Disable gradient computation for efficiency
        for images, labels in test_loader: # Iterate over test dataset batches
            images, labels = images.to(device), labels.to(device) # Move data to GPU/CPU

            outputs = model_ft(images) # Perform forward pass

            loss = loss_function_ft(outputs, labels) # Compute loss
            test_loss += loss.item() * images.size(0) # Accumulate total test loss

            _, prediction = torch.max(outputs.data, 1) # Get predicted class with highest probability
            test_accuracy += int(torch.sum(prediction == labels.data)) # Count correct predictions

            all_test_labels.extend(labels.cpu().numpy()) # Store true labels
            all_test_preds.extend(prediction.cpu().numpy()) # Store predicted labels

    # Compute average accuracy and loss over the entire test dataset
    test_accuracy /= len(test_data)
    test_loss /= len(test_data)

    # Compute additional classification metrics
    test_precision = precision_score(all_test_labels, all_test_preds, average='macro') # Compute precision
    test_recall = recall_score(all_test_labels, all_test_preds, average='macro') # Compute recall
    test_f1 = f1_score(all_test_labels, all_test_preds, average='macro') # Compute F1-score

    # Print test performance metrics
    print(f'Test Accuracy: {test_accuracy:.4f} Test Loss: {test_loss:.4f} '
        f'Test Precision: {test_precision:.4f} Test Recall: {test_recall:.4f} Test F1: {test_f1:.4f}')

    # Print the classification report and confusion matrix for the test set
    print('Classification Report:')
    print(classification_report(all_test_labels, all_test_preds, target_names=classes)) # Generate detailed report



if __name__ == "__main__":
    main()
