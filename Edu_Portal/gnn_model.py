"""
This script defines a simple Graph Neural Network (GNN) model using PyTorch. 
The model is designed for tasks such as recommendation generation or classification. 

Model Architecture:
1. A fully connected layer (fc1) transforms the input data to a hidden space with dimensionality `hidden_dim`.
2. A second fully connected layer (fc2) further reduces the representation to a size of 32.
3. A final output layer (fc3) produces the logits (raw predictions) of size `output_dim`.

Activation Functions:
- ReLU activation is applied after the first two fully connected layers to introduce non-linearity.

The model is suitable for tasks where the output is a set of logits or continuous values without activation.

Dependencies:
- PyTorch (torch) for defining and training the neural network model.
"""



# gnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        
        # Define the layers of the model
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer without activation for logits
        return x
 