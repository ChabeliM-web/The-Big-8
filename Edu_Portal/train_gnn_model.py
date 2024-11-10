"""
This script trains a Graph Neural Network (GNN) to predict school performance based on numeric features. It loads and preprocesses the 
data, splitting it into training and validation sets while standardizing the features. The script defines a GNN model with three layers, 
uses Cross-Entropy Loss, and optimizes with Adam. It trains the model for a set number of epochs, tracking loss and accuracy, then saves 
the trained model to a file. Finally, the script loads the saved model checkpoint to verify its parameters.

"""




# Filename: train_gnn_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

# dataset class
class SchoolDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# GNN model architecture
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer
        return x

# Load data
data = pd.read_csv('data/gauteng_schools.csv')  

# Preprocess data: Select only numeric features
X = data.select_dtypes(include=['float64', 'int64']).values  # Only keep numeric columns
y = data['PERFORMANCE PERCENTAGE 2023'].values  # Your label column


y = pd.cut(y, bins=7, labels=False)  

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Create datasets and dataloaders
train_dataset = SchoolDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = SchoolDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]         
hidden_dim = 64                      
output_dim = len(set(y_train))       

model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=50)

# Save the model
torch.save(model.state_dict(), 'trained_model.pth')
print("Model trained and saved as 'trained_model.pth'")


checkpoint = torch.load('trained_model.pth', map_location=torch.device('cpu'), weights_only=True)
print("Checkpoint loaded. Model parameters:")
for name, param in model.named_parameters():
    print(name, param.size())
