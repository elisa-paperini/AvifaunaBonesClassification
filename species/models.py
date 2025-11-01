import torch.nn as nn

class CustomModule(nn.Module):
    """
    Custom classification head for ResNet.
    
    This module replaces the final fully connected layer of a pre-trained model like ResNet.
    It takes the high-level features from the backbone (e.g., a 2048-dimensional vector)
    and processes them through a series of layers to produce classification scores (logits).
        
    Args:
        num_ftrs (int): Number of input features from the backbone model (e.g., 2048 for ResNet-50/101).
        num_classes (int): The number of classes for the final classification task.
    """
    def __init__(self, num_ftrs, num_classes):
        super().__init__()
        # 2048 -> 512
        self.layer1 = nn.Linear(num_ftrs, 512)
        self.norm1  = nn.BatchNorm1d(512)
        # LeakyReLU activation function
        self.act1   = nn.LeakyReLU()
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # 512 -> 256 -> num_classes
        self.layer_mid = nn.Linear(512, 256)
        self.norm2 = nn.BatchNorm1d(256)
        self.act2      = nn.LeakyReLU()
        self.layer_out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout(x)
        # Feature Extraction
        features = x
        x = self.layer_mid(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        # The final output of the network: raw, unnormalized scores for each class (logits).
        logits = self.layer_out(x)
        
        # Return both the intermediate features and the final logits.
        return features, logits

def init_weights(m):
    """
    Applies a custom weight initialization scheme to the layers of a model.
    It initializes Linear layers using Kaiming (He) initialization and sets standard
    initial values for BatchNorm layers.
    """
    # Check if the module 'm' is an instance of a Linear (fully connected) layer.
    if isinstance(m, nn.Linear):
        # Apply Kaiming (He) normal initialization to the weights
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # Check if the module 'm' is an instance of a 1D Batch Normalization layer.
    elif isinstance(m, nn.BatchNorm1d):
        # Initialize the learnable affine parameter 'weight' (gamma) to 1.
        nn.init.constant_(m.weight, 1)
        # Initialize the learnable affine parameter 'bias' (beta) to 0.
        nn.init.constant_(m.bias, 0)