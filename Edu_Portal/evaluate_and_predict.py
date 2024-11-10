"""
This Python script evaluates the performance of a Graph Neural Network (GNN) model for school recommendation tasks. It first loads user 
review data and school data from CSV files. The script then processes the user data through the trained GNN model, generating predictions 
for each data point. For each prediction, the model outputs the top 7 predicted school IDs. These predictions are compared to the true school
 IDs (from the `user_data`), and several evaluation metrics—precision, recall, F1 score, and Mean Average Precision (MAP)—are calculated to 
 assess the model's accuracy and effectiveness. The evaluation results are printed out, giving a clear view of how well the model performs 
 in terms of these key metrics. The script also uses a random validation set generated based on the user data for testing the model.

"""

import sys
import os
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gnn_model import GNNModel


# E: a dictionary mapping index to school_id
index_to_school_id = {0: 700910011, 1: 700400393, 2: 700121210, 3: 700350561, 4: 700915064}

def evaluate_model(predictions, true_labels):
    y_true = np.array(true_labels)
    
    # Map predicted indices to actual school IDs
    y_pred = np.array([index_to_school_id.get(pred, -1) for pred in [top_k[0] for top_k in predictions]])
    
    print(f"True labels: {y_true[:5]}")
    print(f"Predicted labels: {y_pred[:5]}")
    
    precision = precision_score(y_true, y_pred, average="macro", zero_division=1)
    print(f"Precision: {precision}")

    recall = recall_score(y_true, y_pred, average="macro", zero_division=1)
    print(f"Recall: {recall}")

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=1)
    print(f"F1 Score: {f1}")

    map_score = np.mean([
        precision_score(y_true == i, y_pred == i, zero_division=1)
        for i in range(y_true.max() + 1)
    ])
    print(f"MAP Score: {map_score}")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "MAP": map_score
    }




if __name__ == "__main__":
    user_data = pd.read_csv('data/generate_user_reviews_with_ids.csv')
    school_data = pd.read_csv('data/gauteng_schools.csv')

    true_labels = user_data['school_id'].tolist()  # Extract true labels

    # Set input dimension based on user_data 
    input_dim = 2

    # Generate random validation data with the correct input dimension
    validation_data = [torch.randn(1, input_dim) for _ in range(len(user_data))]

    output_dim = 7
    model = GNNModel(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
    model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu')))
    model.eval()

    print("Generating Predictions...")
    predictions = []
    with torch.no_grad():
        for i, data_point in enumerate(validation_data):
            output = model(data_point)
            top_k_predictions = torch.topk(output, k=7, dim=1).indices.squeeze().tolist()
            predictions.append(top_k_predictions)
            
            # Print progress for every data point processed
            print(f"Processed {i+1} data points")


    metrics = evaluate_model(predictions, true_labels)

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
