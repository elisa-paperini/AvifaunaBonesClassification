# Import standard libraries for data manipulation, file handling, and timing
import copy  # For deep copying model weights
import numpy as np  # For numerical operations
import time  # To time operations
import os  # For interacting with the operating system (e.g., file paths)
from functools import partial  # To pre-fill function arguments, used with Optuna
import itertools  # For creating iterators for efficient looping
from PIL import ImageFile  # From Pillow library for image handling
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allows loading of truncated image files

# Import PyTorch and TorchVision for deep learning
import torch  # Main PyTorch library
import torch.nn as nn  # Neural network modules
import torch.utils.data  # Data handling utilities
import torch.optim as optim  # Optimization algorithms (e.g., SGD)
import torchvision  # Computer vision library for PyTorch
import torchvision.transforms as transforms  # For image transformations and augmentations
from torchvision import models  # For standard datasets and pre-trained models
from torch.optim import lr_scheduler  # For adjusting the learning rate during training

# Import scikit-learn libraries for machine learning and evaluation
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation metrics
from sklearn.model_selection import train_test_split  # For splitting data

# Import Optuna for hyperparameter optimization
import optuna

# Import custom modules from the local directory
from models import CustomModule, init_weights  # Custom model head and weight initialization
from data import N_AugmentedDataset  # Custom dataset wrapper for augmentations
from utils import objective  # Utility functions



def main():
    '''
    **Taxonomic Classification with ResNet and Machine Learning**
    Workflow: Optuna hyperparameter tuning, CNN training with fine-tuning, 
    visualizations related to CNN, final 
    Machine Learning classification on extracted features.
    '''

    #----- DEVICE SETUP -----#
    # Set the computation device to CUDA (GPU) if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()  # Clear any cached memory on the GPU
    print('------------------------')
    print('Selected device:', device)
    print('------------------------')
    print(' ')
    since = time.time()  # Start a timer for the entire process



    #----- DATA PATHS -----#
    # Define the data directories (*change with your actual path*)
    data_dir = 'F:\\elisa\\dataset_ossa\\bones_detection_species_BN'
    training_dir = os.path.join(data_dir, 'train')
    validation_dir =  os.path.join(data_dir, 'val')

    #----- MAIN CONFIGURATION PARAMETERS -----#
    # Load a pre-trained ResNet-101 model with default weights from TorchVision
    model = models.resnet101(weights='DEFAULT')
    # Number of augmented views to create for each training image.
    n_aug_views = 1  # Set to 1 for simple training, can be increased for more robust augmentation.
    # Number of trials for the Optuna hyperparameter search.
    n_trials = 10
    epoch = 500 # Final number of training epochs
    size_Resize = 512

    #----- DATA PRE-PROCESSING & DATASETS -----#
    
    # Define the sequence of transformations for the training data.
    # This includes resizing, data augmentation (rotation, color jitter, blur), and normalization.
    data_transforms_t = transforms.Compose([
                        transforms.Resize(size=(size_Resize,size_Resize)),
                        transforms.RandomRotation(5),
                        transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=0, hue=0.1),
                        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.05, 0.3)),
                        transforms.ToTensor(),  # Convert PIL image to a PyTorch tensor
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
                        ])

    # Define the sequence of transformations for the validation data.
    # This only includes resizing and normalization, no data augmentation.
    data_transforms_v = transforms.Compose([
                        transforms.Resize(size=(size_Resize,size_Resize)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    # Create PyTorch datasets using ImageFolder, which automatically finds class folders.
    base_training_dset = torchvision.datasets.ImageFolder(training_dir, data_transforms_t)
    validation_dset = torchvision.datasets.ImageFolder(validation_dir, data_transforms_v)
    class_names_tr = base_training_dset.classes  # Get the list of class names from the dataset




    #----- MODEL CUSTOMIZATION -----#
    # Get the number of input features for the original fully connected layer of ResNet
    num_ftrs = model.fc.in_features  # For ResNet-101, this is 2048
    
    # Replace the final fully connected layer with CustomModule
    model.fc = CustomModule(num_ftrs, len(class_names_tr)).to(device)
    
    # Freezing/Fine-tuning setup
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last convolutional block (layer4) for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Unfreeze the second to last convolutional block (layer3) as well for more flexibility
    for param in model.layer3.parameters():
        param.requires_grad = True
    
    # Unfreeze the custom head (model.fc)
    for param in model.fc.parameters():
        param.requires_grad = True




    # ----- OPTUNA HYPERPARAMETER SEARCH -----#
    # Create an Optuna study to find the best hyperparameters. 'maximize' means we want to maximize the objective (accuracy).
    study = optuna.create_study(direction='maximize')
    # Start the optimization process.
    # `partial` is used to pass fixed arguments (like model, datasets) to the `objective` function,
    # while Optuna varies the hyperparameters (lr, batch_size, etc.).
    study.optimize(
        partial(objective, model=model, base_training_dset=base_training_dset, 
                validation_dset=validation_dset, n_aug_views=n_aug_views, device=device), n_trials=n_trials)
    print("\n------------------------")
    print("\nBest hyperparameters:", study.best_params)
    print("Best validation accuracy from Optuna trials:", study.best_value)

    





    # --- FINAL CNN TRAINING & OPTIMAL HYPERPARAMETERS ---
    
    # Set up hyperparameters from Optuna results
    best_params = study.best_params
    lr = best_params['lr']
    batch_size = best_params['batch_size']
    momentum = best_params['momentum']


    # Initialize variables to track the best model performance
    best_acc_top1 = 0.0  # Best top-1 validation accuracy
    best_acc_top3 = 0.0  # Best top-3 validation accuracy
    best_acc_top5 = 0.0  # Best top-5 validation accuracy
    best_acc_ml = 0.0  # Placeholder for machine learning model accuracy (if used later)
    best_model_wts = None  # To store the weights of the best performing model

    # Get sampler for class balancing
    targets = base_training_dset.targets
    class_counts = np.bincount(targets)
    num_samples = len(base_training_dset)
    weights_per_class = num_samples / class_counts
    sample_weights = torch.DoubleTensor([weights_per_class[t] for t in targets])
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # DataLoaders using N_AugmentedDataset
    training_dset = N_AugmentedDataset(base_training_dset, n_aug=n_aug_views)
    # The training loader uses the balanced sampler and drops the last incomplete batch.
    training_loader = torch.utils.data.DataLoader(training_dset, batch_size=batch_size, sampler=sampler, num_workers=2, drop_last=True)
    # The validation loader shuffles data but does not need a sampler.
    validation_loader = torch.utils.data.DataLoader(validation_dset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Re-initialize model head and define optimizer/criterion
    model.fc.apply(init_weights)
    model = model.to(device)
    
    # Optimizer with separate LR for head (lr) and backbone (lr * 0.2)
    optimizer = optim.SGD([
        {'params': model.fc.parameters(), 'lr': lr},
        # Apply the reduced learning rate to both unfrozen backbone layers
        {'params': itertools.chain(model.layer3.parameters(), model.layer4.parameters()), 'lr': lr * 0.2}
    ], momentum=momentum, weight_decay=1e-4)
    # A learning rate scheduler to decrease the LR by a factor of 0.1 every 20 epochs.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # The loss function for multi-class classification.
    criterion = nn.CrossEntropyLoss()

    # --- Main Training and Validation Loop ---
    for e in range(epoch):
        model.train()  # Set the model to training mode
        # Initialize trackers for training metrics for the current epoch
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        total = 0
        loss_running = 0
        all_train_preds = []
        all_train_labels = []

        
        # --- Training Phase ---
        for inputs, labels in training_loader:
            # Reshape inputs from (Batch, N_views, C, H, W) to (Batch*N_views, C, H, W)
            inputs = inputs.view(-1, inputs.size(-3), inputs.size(-2), inputs.size(-1))
            labels = labels.repeat(n_aug_views) 
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass: get features and final output logits
            features, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Tracking stats
            _, top1_preds = torch.max(outputs, 1)
            _, top3_preds = torch.topk(outputs, 3, dim=1)
            _, top5_preds = torch.topk(outputs, 5, dim=1)
            loss_running += loss.item() * inputs.size(0)
            correct_top1 += (top1_preds == labels).sum().item()
            correct_top3 += (top3_preds == labels.view(-1, 1)).any(dim=1).sum().item()
            correct_top5 += (top5_preds == labels.view(-1, 1)).any(dim=1).sum().item()
            total += labels.size(0)

            # Store predictions and labels for classification report
            all_train_preds.extend(top1_preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        # Calculate average loss and accuracy for the training epoch
        epoch_loss = loss_running / total
        epoch_acc_top1 = correct_top1 / total
        epoch_acc_top3 = correct_top3 / total
        epoch_acc_top5 = correct_top5 / total
        scheduler.step()

        # --- Validation Phase ---
        model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
        # Initialize trackers for validation metrics
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        total = 0
        loss_current_val = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():  # Disable gradient calculations for validation
            for val_inputs, val_labels in validation_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_features, val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                loss_current_val += val_loss.item() * val_inputs.size(0)
                
                # Top-1, Top-3 and Top-5 accuracy
                _, top1_preds = torch.max(val_outputs, 1)
                _, top3_preds = torch.topk(val_outputs, 3, dim=1)
                _, top5_preds = torch.topk(val_outputs, 5, dim=1)
                correct_top1 += (top1_preds == val_labels).sum().item()
                correct_top3 += (top3_preds == val_labels.view(-1, 1)).any(dim=1).sum().item()
                correct_top5 += (top5_preds == val_labels.view(-1, 1)).any(dim=1).sum().item()
                total += val_labels.size(0)

                # Store predictions and labels for classification report
                all_val_preds.extend(top1_preds.cpu().numpy())
                all_val_labels.extend(val_labels.cpu().numpy())
                
        # Calculate average loss and accuracy for the validation epoch
        val_epoch_loss = loss_current_val / total
        val_acc_top1 = correct_top1 / total
        val_acc_top3 = correct_top3 / total
        val_acc_top5 = correct_top5 / total
        
        # --- Check for Best Model ---
        # If the current validation accuracy is the best so far, save the model weights
        if val_acc_top1 > best_acc_top1:
            best_acc_top1 = val_acc_top1
            best_acc_top3 = val_acc_top3 # Store top-3 acc when top-1 is best
            best_acc_top5 = val_acc_top5 # Store top-5 acc when top-1 is best
            best_model_wts = copy.deepcopy(model.state_dict())   

        print(' ')
        print('------------------------')
        print('CNN training epoch:', (e+1))
        print('CNN Training:   Loss {:.4f}, Top-1 Acc {:.4f}, Top-3 Acc {:.4f}, Top-5 Acc {:.4f}'.format(epoch_loss, epoch_acc_top1, epoch_acc_top3, epoch_acc_top5))
        print('CNN Validation: Loss {:.4f}, Top-1 Acc {:.4f}, Top-3 Acc {:.4f}, Top-5 Acc {:.4f}'.format(val_epoch_loss, val_acc_top1, val_acc_top3, val_acc_top5))
        print("CNN Best Top-1 Accuracy: {:.4f} (Top-3: {:.4f}, Top-5: {:.4f})".format(best_acc_top1, best_acc_top3, best_acc_top5))
        
        print("\nTraining Classification Report (Epoch {}):".format(e+1))
        print(classification_report(all_train_labels, all_train_preds, target_names=class_names_tr, zero_division=0))
        print("\nValidation Classification Report (Epoch {}):".format(e+1))
        print(classification_report(all_val_labels, all_val_preds, target_names=class_names_tr, zero_division=0))



    # --- SAVE THE BEST MODEL TO DISK ---
    if best_model_wts:
        torch.save(best_model_wts, 'best_cnn_model.pth')
        print("\n------------------------")
        print("Best CNN model weights saved to 'best_cnn_model.pth'")
        print("-----------------------\n")
        
    # Load the best CNN weights based on validation accuracy
    if best_model_wts is not None:
        print('\n------------------------')
        print('Loading best CNN model weights for feature extraction...')
        model.load_state_dict(best_model_wts)
        print('------------------------\n')
    model.eval() 



if __name__ == "__main__":
    main()
