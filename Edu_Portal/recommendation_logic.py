"""
This file uses a trained Graph Neural Network (GNN) model to generate school recommendations based on user embeddings,
which represent user preferences in numerical form. The `generate_recommendations` function applies the GNN model to 
the input user embedding, using `torch.no_grad()` to prevent the model from tracking gradients, as this is only an 
inference process. In the main script, the trained GNN model is loaded, initialized with specific input and output 
dimensions, and set to evaluation mode to ensure efficient performance. A sample user embedding is then provided to 
demonstrate the recommendation generation process, and the output recommendations are printed. This approach allows 
the GNN model to suggest schools tailored to individual user profiles.
"""



import torch
from models.gnn_model import GNNModel

def generate_recommendations(user_embedding, model):
    """Generate school recommendations based on user embeddings."""
    with torch.no_grad():
        recommendations = model(user_embedding)
    return recommendations

if __name__ == "__main__":
    # Load the trained model
    model = GNNModel(input_dim=1433, hidden_dim=64, output_dim=7)  # Adjust dimensions based on your model
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()  # Set the model to evaluation mode

    user_embedding = torch.randn(1, 1433)  # Replace with actual user embedding
    recommendations = generate_recommendations(user_embedding, model)
    print("Recommendations:", recommendations)
