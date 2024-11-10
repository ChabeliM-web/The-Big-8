"""
This script implements a Graph Neural Network (GNN)-based recommendation system using a Tkinter GUI. 
The application allows users to generate school recommendations based on user reviews using a pre-trained model. 

Key steps in the process:
1. Loading user and school data.
2. Preprocessing the data for model use.
3. Initializing and loading the pre-trained GNN model.
4. Making predictions using the model.
5. Evaluating the model's performance using various metrics (precision, recall, F1 score).
6. Generating and displaying recommendations for a sample user.

The user interface includes a button to trigger the recommendation process and a scrollable text box to display logs and results.

External Dependencies:
- gnn_model: Contains the definition of the GNN model.
- data_preparation: Functions to load and preprocess the data.
- evaluate_and_predict: Functions to evaluate the model's predictions.
- recommender: Function to generate recommendations based on a user embedding.
"""



import sys
import os
import torch
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import threading

# Importing necessary functions from other files
from gnn_model import GNNModel
from data_preparation import load_user_data, load_school_data, preprocess_data
from evaluate_and_predict import evaluate_model
from recommender import generate_recommendations

# Set paths and constants
USER_DATA_PATH = 'data/generate_user_reviews_with_ids.csv'
SCHOOL_DATA_PATH = 'data/gauteng_schools.csv'
MODEL_PATH = 'trained_model.pth'

# E: a dictionary mapping index to school_id
index_to_school_id = {0: 700910011, 1: 700400393, 2: 700121210, 3: 700350561, 4: 700915064}

class RecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GNN Recommendation System")
        self.root.geometry("600x400")

        # Initialize data and model variables
        self.user_data = None
        self.school_data = None
        self.model = None
        self.true_labels = None
        self.validation_data = None

        # Create the GUI layout
        self.create_widgets()

    def create_widgets(self):
        # Create a frame for the display screen and button
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20, padx=20, expand=True, fill="both")

        # Create the display screen (Text Box) to show logs
        self.output_box = scrolledtext.ScrolledText(main_frame, width=70, height=15, wrap=tk.WORD)
        self.output_box.pack(pady=10, expand=True)

        # Create "Generate Recommendations" button
        self.recommend_btn = tk.Button(main_frame, text="Generate Recommendations", command=self.on_generate_button_click)
        self.recommend_btn.pack(pady=10)

    def on_generate_button_click(self):
        """Start the recommendation generation in a separate thread."""
        self.recommend_btn.config(state=tk.DISABLED)  # Disable the button to prevent multiple clicks
        self.output_box.delete(1.0, tk.END)  # Clear previous logs
        self.output_box.insert(tk.END, "Processing data and generating recommendations...\n")

        # Start the background thread to run the long task
        thread = threading.Thread(target=self.generate_recommendations)
        thread.start()

    def generate_recommendations(self):
        """Generate recommendations and display log."""
        try:
            # Load user and school data
            self.update_log("Loading user and school data...")
            self.user_data = load_user_data(USER_DATA_PATH)
            self.school_data = load_school_data(SCHOOL_DATA_PATH)

            if self.user_data is None or self.school_data is None:
                raise ValueError("Data loading failed. Please check your files.")

            self.update_log("Data loaded successfully.")

            # Preprocess data
            self.update_log("Preprocessing data...")
            self.user_data, self.school_data = preprocess_data(self.user_data, self.school_data)
            self.true_labels = self.user_data['school_id'].tolist()
            self.validation_data = [torch.randn(1, 2) for _ in range(len(self.user_data))]

            # Initialize the model
            self.update_log("Initializing the model...")
            input_dim = 2  # Adjust according to your feature size
            output_dim = 7
            self.model = GNNModel(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            self.model.eval()  # Set the model to evaluation mode
            self.update_log("Model initialized successfully.")

            # Generate predictions
            self.update_log("Generating predictions...")
            predictions = []
            with torch.no_grad():
                for i, data_point in enumerate(self.validation_data):
                    output = self.model(data_point)
                    top_k_predictions = torch.topk(output, k=7, dim=1).indices.squeeze().tolist()
                    predictions.append(top_k_predictions)

            # Evaluate model
            self.update_log("Evaluating model...")
            metrics = evaluate_model(predictions, self.true_labels)
            self.update_log("Model Evaluation Results:")
            for metric, value in metrics.items():
                self.update_log(f"{metric}: {value:.4f}")

            # Generate recommendations for a sample user
            self.update_log("Generating recommendations for a sample user...")
            user_embedding = torch.randn(1, 2)  # Replace with actual user embedding
            recommendations = generate_recommendations(user_embedding, self.model)
            self.update_log("\nSample Recommendations:")
            self.update_log(str(recommendations))

        except Exception as e:
            self.update_log(f"Error: {str(e)}")

        finally:
            self.recommend_btn.config(state=tk.NORMAL)  # Re-enable the button

    def update_log(self, message):
        """Update the log text box with a new message."""
        self.output_box.insert(tk.END, message + "\n")
        self.output_box.yview(tk.END)  # Scroll to the bottom

# Create the main window and pass it to the App
root = tk.Tk()
app = RecommendationApp(root)
root.mainloop()
