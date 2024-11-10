"""
This script trains an enhanced neural network-based chatbot model to generate responses based on user questions.

Steps:
1. Data preprocessing includes loading the dataset, encoding labels, and vectorizing text using TF-IDF.
2. The data is split into training and validation sets.
3. A neural network model is defined with layers such as fully connected layers, batch normalization, dropout for regularization, and LeakyReLU activation.
4. The model is trained with a weighted loss function to handle class imbalance.
5. Early stopping is implemented to avoid overfitting, saving the best model based on validation loss.
6. The final model, along with the LabelEncoder and TF-IDF Vectorizer, are saved for future use.

Dependencies:
- torch for model training
- scikit-learn for data preprocessing and evaluation
- pickle for saving the vectorizer

Parameters:
- The dataset should contain 'question' and 'answer' columns, where 'question' is the input and 'answer' is the output.
- The model is trained with 50 epochs, early stopping with patience of 3, and a learning rate scheduler for adaptive learning rates.

Returns:
- Saved model ('chatbot_model_final.pth')
- Saved LabelEncoder classes ('label_encoder_classes.npy')
- Saved TF-IDF Vectorizer ('vectorizer.pkl')
"""




#train_chatbot_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pickle

# Load and preprocess the dataset
df = pd.read_csv('chatbot_training_data_expanded_v3.csv')
X = df['question'].values
y = df['answer'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Vectorize text data using TF-IDF with increased max features
vectorizer = TfidfVectorizer(max_features=2000)
X_vectorized = vectorizer.fit_transform(X).toarray()

# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

# Convert data to tensors and use DataLoader for batching
X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).long()
X_val_tensor = torch.tensor(X_val).float()
y_val_tensor = torch.tensor(y_val).long()

train_data = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
val_data = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32)

# Calculate class weights for imbalance handling
class_weights = 1. / np.bincount(y_train)
weights = torch.tensor(class_weights, dtype=torch.float32)

# Define an enhanced neural network model
class EnhancedChatbotModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(0.4)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return self.softmax(x)

# Initialize model, loss function, optimizer, and learning rate scheduler
input_dim = X_train.shape[1]
output_dim = len(np.unique(y_encoded))
model = EnhancedChatbotModel(input_dim, output_dim)

loss_fn = nn.CrossEntropyLoss(weight=weights)  # Weighted loss function for imbalance
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Early stopping parameters
early_stopping_patience = 3
best_val_loss = float('inf')
patience_counter = 0

# Training loop with early stopping
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_data:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_X, batch_y in val_data:
            val_outputs = model(batch_X)
            val_loss += loss_fn(val_outputs, batch_y).item()
            predictions = torch.argmax(val_outputs, dim=1)
            correct += (predictions == batch_y).float().sum().item()

    val_loss /= len(val_data)
    val_accuracy = correct / len(X_val_tensor)

    # Adjust learning rate
    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'chatbot_model_best.pth')
    else:
        patience_counter += 1

    # Print progress
    print(
        f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss/len(train_data):.4f}, "
        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
    )

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break

# Save the final model, LabelEncoder, and TF-IDF Vectorizer
torch.save(model.state_dict(), 'chatbot_model_final.pth')
np.save('label_encoder_classes.npy', label_encoder.classes_)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Enhanced Model, LabelEncoder, and Vectorizer saved successfully!")
print("Chatbot training completed successfully!")

