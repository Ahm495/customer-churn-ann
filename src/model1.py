import torch
import torch.nn as nn

class ChurnModel(nn.Module):
    def __init__(self, input_size):
        super(ChurnModel, self).__init__()

        # First Layer: Expanded to 128 units for better feature extraction
        self.layer1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        # Second Layer: 64 units
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Third Layer: 32 units
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        
        # Fourth Layer (Additional): 16 units to refine features
        self.layer4 = nn.Linear(32, 16)
        
        # Output Layer: Single unit for binary classification
        self.output = nn.Linear(16, 1)

        # Helper layers and activations
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Increased Dropout to 0.3 to prevent overfitting in a deeper network
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Layer 1: Linear -> BN -> ReLU -> Dropout
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)

        # Layer 2: Linear -> BN -> ReLU -> Dropout
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)

        # Layer 3: Linear -> BN -> ReLU -> Dropout
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)

        # Layer 4: Linear -> ReLU
        x = self.relu(self.layer4(x))

        # Output Layer: Sigmoid for probability mapping
        x = self.sigmoid(self.output(x))
        return x