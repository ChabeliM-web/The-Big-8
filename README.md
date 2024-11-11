<!DOCTYPE html>
<html lang="en">

<body>

<h1>EduPortal</h1>

<p><strong>EduPortal</strong> - This AI Solution project is designed to help people find the best-fit schools in Gauteng if they’re looking to enroll. Using artificial intelligence, it carefully examines various details about each school—such as academic performance, location, grade level, and feedback from others—to provide personalized recommendations. By applying techniques like natural language processing (NLP) to interpret what users are looking for, and using graph neural networks (GNNs) to understand connections between schools, districts, and users, the system offers recommendations that are relevant and insightful. The goal of this project is to simplify the school selection process, making it easier and faster for users to find schools that meet their needs.
</p>

<h2>Features</h2>
<ul>
    <li><strong>AI-Powered School Recommender:</strong> An AI-driven system that provides tailored school recommendations based on user preferences and data patterns.</li>
    <li><strong> Chatbot system:</strong> An interactive tool that uses AI to respond to user queries, offering personalized information and assistance.</li>
    <li><strong>Natural Language Processing:</strong> A field of AI focused on enabling machines to understand, interpret, and generate human language.</li>
    <li><strong>Speech Synthisis:</strong> Technology that converts text into spoken language, allowing systems to communicate with users via audio.</li>
    
   
</ul>

<h2>Installation</h2>
<ol>
    <li>Clone the repository:
        <pre><code>git clone https://github.com/ChabeliM-web/The-Big-8.git</code></pre>
    </li>
    <li>Navigate to the project directory:
        <pre><code>cd EduPortalCBS</code></pre>
    </li>
    <li>Prepare your datasets
        <ul>
            <li><code>gauteng_schools.csv</code></li>
            <li><code>generate_user_reviews.csv</code></li>
        </ul>
    </li>
    <li>Train your models</code>.</li>
        <ul>
            <li><code> train_gnn_model</code></li>
            <li><code>train_chatbot_nodel</code></li>
        </ul>
    <li>Run your main file:
        <ul>
            <li><code>main.py</code></li>
        </ul>
    </li>
    <li>Run Chatbot file</li>
    <ul>
        <li><code> chat_bot.py </code></li>
    </ul>
</ol>

<h2>Files and Directories</h2>
    <li><code>data_preparation.py<code>: This script loads and preprocesses user and school review data from CSV files, handling missing values, displaying data samples, and returning the cleaned data for further use.</li>
    <li><code>evaluate_and_predict.py</code>: This Python script evaluates a Graph Neural Network (GNN) model's performance for school recommendations by generating top 7 school predictions from user review data and calculating precision, recall, F1 score, and Mean Average Precision (MAP) for model accuracy.</li>
    <li><code>generate_user_reviews_with_ids.py</code>: This script generates random reviews and ratings for a set of schools, storing the simulated user feedback in a CSV file for use in training recommendation models or data analysis. </li>
    <li><code>generate_user_reviews_with_ids.csv</code>: Dataset containing user reviews</li>
    <li><code>recommendation_logic.py</code>: This file generates school recommendations by applying a trained Graph Neural Network (GNN) model to user preference embeddings, providing personalized suggestions based on individual user profiles.</li>
    <li><code>recommender.py</code>: This function uses a trained model to generate school recommendations for a user based on their embedding, returning recommendations without computing gradients to save memory and computation time.</li>
    <li><code>gnn_model.py</code>: This script defines a simple Graph Neural Network (GNN) model using PyTorch with a three-layer fully connected architecture and ReLU activation, designed for recommendation or classification tasks.</li>
    <li><code>chat_bot.py</code>: This Python script creates a chatbot using a PyTorch-based deep learning model for intent classification, which vectorizes user input with a pre-trained TF-IDF vectorizer, predicts responses through a neural network, decodes the output with a LabelEncoder, and uses `pyttsx3` for text-to-speech to provide spoken responses.</li>
    <li><code>chatbot_dataset.py</code>:This Python script expands a chatbot's training dataset by generating additional question-answer pairs, introducing variations to the questions, and saving the expanded dataset to a CSV file for improved chatbot training.</li>
    <li><code>label_encoder_classes.npy</code>:The `label_encoder_classes.npy` file stores the classes (labels) used by a LabelEncoder, which maps categorical values to numerical values for model training and prediction.</li>
        <li><code>gauteng_schools.py</code>:This code creates a DataFrame from a dictionary containing various school details in Gauteng and saves it as a CSV file named 'gauteng_schools.csv' for further analysis or use.This code creates a DataFrame from a dictionary containing various school details in Gauteng and saves it as a CSV file named 'gauteng_schools.csv' for further analysis or use.</li>
        <li><code>gauteng_schools.csv</code>:Dataset containing schools and their contact details for more information</li>
    <li><code>train_gnn_model.py</code>: This script trains a Graph Neural Network (GNN) to predict school performance using numeric features, preprocesses the data, defines the model, and optimizes it, then saves and verifies the trained model.</li>
    <li><code>trained_model.pth</code>: The `trained_model.pth` file contains the weights and parameters of a trained machine learning model, saved for later use in making predictions or further training.</li>
    <li><code>train_chatbot_model.py</code>: This script trains an enhanced neural network-based chatbot model to generate responses, including data preprocessing, model training with a weighted loss function, and saving the final model and necessary components for future use.</li>
    <li><code>chatbot_model.pth</code>: The `chatbot_model.pth` file stores the trained parameters of a deep learning model used by the chatbot to generate responses based on user input.</li>
    <li><code>chatbot_model_best.pth</code>:The `chatbot_model_best.pth` file stores the trained parameters of a deep learning model used by the chatbot to generate responses based on user input.</li>
    <li><code>chatbot_model_final.pth </code>:The `chatbot_model_final.pth` file stores the trained parameters of a deep learning model used by the chatbot to generate responses based on user input.</li>
    <li><code>main.py</code>:This script implements a Graph Neural Network-based recommendation system with a Tkinter GUI, allowing users to generate school recommendations by loading and preprocessing data, making predictions with a pre-trained model, evaluating performance, and displaying results in the interface.</li>
    <li><code>vectorizer.pkl </code>:The `vectorizer.pkl` file contains a pre-trained TF-IDF vectorizer used to transform text data into numerical features for input into a machine learning model.</li>
    
<h2>License</h2>
<p>This project is licensed under the MIT License.</p>

</body>
</html>
