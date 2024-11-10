"""
This Python script implements a chatbot that uses a deep learning model built with PyTorch for intent classification. It first vectorizes 
the user input using a pre-trained TF-IDF vectorizer and then passes the vectorized input through a neural network to make predictions. 
The output from the model is decoded using a LabelEncoder, which maps the predicted label to a meaningful response. The chatbot integrates 
text-to-speech (TTS) functionality via the `pyttsx3` library, allowing it to speak its responses aloud. Users interact with the chatbot by 
typing their input in the terminal, and the chatbot responds both in text and via voice. 

"""


import torch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pyttsx3

# Define the neural network model
class EnhancedChatbotModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedChatbotModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5 = torch.nn.Linear(32, output_dim)
        self.dropout = torch.nn.Dropout(0.4)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return self.softmax(x)

# Load the pre-trained TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

# Initialize the chatbot model and load weights
input_dim = 116  # Assuming this was your original input size for TF-IDF
output_dim = len(label_encoder.classes_)
model = EnhancedChatbotModel(input_dim, output_dim)
model.load_state_dict(torch.load('chatbot_model_best.pth', weights_only=True))
model.eval()

# Initialize the speech engine for text-to-speech
engine = pyttsx3.init()

# Function to get response based on TF-IDF vectorization
def get_response(user_input):
    # Vectorize the user input
    user_input_vectorized = vectorizer.transform([user_input]).toarray()
    input_tensor = torch.tensor(user_input_vectorized).float()

    # Perform prediction using the model
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get predicted label
    predicted_index = torch.argmax(output, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_label

# Function to make the chatbot speak (voice-over response)
def speak(response):
    engine.say(response)
    engine.runAndWait()

# Main loop for user interaction
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        speak("Goodbye!")  
        break
    response = get_response(user_input)
    print(f"Chatbot: {response}")
    speak(response)
