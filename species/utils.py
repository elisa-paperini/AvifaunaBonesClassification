# Import PyTorch and related libraries for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
# Import scikit-learn components for machine learning and evaluation
from sklearn.metrics import accuracy_score
# Import other standard libraries
import numpy as np
import itertools
# Import custom local modules
from data import N_AugmentedDataset
from models import init_weights



def train_and_eval_cnn(model, training_loader, validation_loader, lr, momentum, n_aug_views, device, epoch_objective):
    """
    Performs the CNN training and validation loop. 
    This is a reusable core function used by both the Optuna objective and the final training run.
    
    Returns:
        tuple: (best_validation_accuracy, best_model_weights)
    """
    # Re-initialize the model's head. This is crucial for Optuna to ensure each trial starts fresh.
    model.fc.apply(init_weights)
    model = model.to(device)
    
    # Set up the optimizer with different learning rates for the head and the fine-tuned backbone layers.
    optimizer = optim.SGD([
        {'params': model.fc.parameters(), 'lr': lr},
        # Apply the reduced learning rate to both unfrozen backbone layers
        {'params': itertools.chain(model.layer3.parameters(), model.layer4.parameters()), 'lr': lr * 0.2}
    ], momentum=momentum, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_wts = None

    # --- Main Training and Validation Loop ---
    for e in range(epoch_objective):
        model.train()
        # Training loop
        for inputs, labels in training_loader:
            # Reshape augmented data and move to the computation device
            inputs = inputs.view(-1, inputs.size(-3), inputs.size(-2), inputs.size(-1))
            labels = labels.repeat(n_aug_views)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            features, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode
        correct_top1 = 0
        correct_top3 = 0
        total = 0
        with torch.no_grad():  # No gradients needed for validation
            for val_inputs, val_labels in validation_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                
                val_features, val_outputs = model(val_inputs)
                
                # Top-1 prediction
                _, top1_preds = torch.max(val_outputs, 1)
                correct_top1 += (top1_preds == val_labels).sum().item()
                # Top-3 prediction
                _, top3_preds = torch.topk(val_outputs, 3, dim=1)
                correct_top3 += (top3_preds == val_labels.view(-1, 1)).any(dim=1).sum().item()
                total += val_labels.size(0)
        
        val_acc = correct_top1 / total
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Only store weights if running final train, otherwise not needed for Optuna
            best_model_wts = model.state_dict() 

    return best_acc, best_model_wts


def objective(trial, model, base_training_dset, validation_dset, n_aug_views, device):
    """
    Optuna objective function for hyperparameter search (finding best LR, BS, Momentum).
    """
    # Hyperparameter suggestion
    lr = trial.suggest_float('lr', 0.001, 0.2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    momentum = trial.suggest_float('momentum', 0, 0.8)
    epoch_objective = 5 # Run for fewer epochs for faster Optuna search

    # Sampler setup for class balancing
    targets = base_training_dset.targets
    class_counts = np.bincount(targets)
    num_samples = len(base_training_dset)
    weights_per_class = num_samples / class_counts
    sample_weights = torch.DoubleTensor([weights_per_class[t] for t in targets])
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    training_dset = N_AugmentedDataset(base_training_dset, n_aug=n_aug_views)
    training_loader = torch.utils.data.DataLoader(training_dset, batch_size=batch_size, sampler=sampler, num_workers=2, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_dset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Train and evaluate, but don't save model weights here
    best_acc, _ = train_and_eval_cnn(
        model=model, training_loader=training_loader, validation_loader=validation_loader,
        lr=lr, momentum=momentum, n_aug_views=n_aug_views, device=device, epoch_objective=epoch_objective
    )
    
    return best_acc