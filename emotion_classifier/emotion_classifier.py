#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Emotion Classifier Model

This script:
1. Trains the best text-only model for emotion classification
2. Saves the trained model to a file
3. Provides functionality to classify emotions in new text input

Usage:
    python emotion_classifier.py
    
After running this script once to train and save the model, you can use it to
classify emotions in new sentences directly.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path - this should fix the import issues
project_root = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules after setting up path correctly
from emotion_classifier.data import load_raw_data, prepare_dataset_splits
from emotion_classifier.preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# Label mapping for consistent classification
LABEL_MAPPING = {"Negative": 0, "Ambiguous": 1, "Neutral": 2, "Mixed": 3, "Positive": 4}
INV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# File paths
MODEL_DIR = os.path.join(project_root, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "text_only_model.pkl")


def preprocess_corpus(texts):
    """
    Apply preprocessing pipeline to a corpus of texts.
    
    Args:
        texts (list): List of text strings to preprocess
        
    Returns:
        list: List of preprocessed text strings
    """
    processed_texts = []
    for text in texts:
        tokens = preprocess_text(
            text,
            lower=True,
            decontract=True,
            remove_punct=True, 
            remove_digits=True,
            remove_stops=True,
            stemming=False,
            lemmatization=True
        )
        processed_texts.append(" ".join(tokens))
    return processed_texts


def build_and_train_model():
    """
    Build and train the best text-only model for emotion classification.
    
    Returns:
        Pipeline: Trained preprocessing and model pipeline
    """
    # Load data
    print("Loading and preparing dataset...")
    train_df, test_df = load_raw_data()
    
    try:
        train_df, val_df, test_df = prepare_dataset_splits(train_df)
        print(f"Created splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    except Exception as e:
        print(f"Using existing splits: {e}")
    
    # Preprocess text data
    print("Preprocessing text data...")
    train_processed = preprocess_corpus(train_df['text'].tolist())
    
    # Create TF-IDF features (the best text-only feature representation)
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(min_df=2, max_features=5000)
    X_train = tfidf_vectorizer.fit_transform(train_processed)
    
    # Get target labels
    y_train = train_df['sentiment'].map(LABEL_MAPPING).values
    
    # Create and train MLP model (best performing text-only model)
    print("Training MLP (Neural Network) model...")
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Create a pipeline for easy inference
    pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('classifier', model)
    ])
    
    return pipeline


def save_model(model):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained model to save
    """
    # Create directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    """
    Load a trained model from file.
    
    Returns:
        Object: Loaded model
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Training new model...")
        model = build_and_train_model()
        save_model(model)
        return model
    
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def classify_emotion(text, model=None):
    """
    Classify the emotion of a given text.
    
    Args:
        text (str): Text to classify
        model (optional): Pre-loaded model to use
        
    Returns:
        str: Predicted emotion label
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Make prediction
    prediction = model.predict([text])[0]
    
    # Map prediction to label
    emotion = INV_LABEL_MAPPING[prediction]
    
    return emotion


def predict_with_probabilities(text, model=None):
    """
    Classify the emotion of a given text with probabilities.
    
    Args:
        text (str): Text to classify
        model (optional): Pre-loaded model to use
        
    Returns:
        tuple: (predicted emotion, probability dictionary)
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Get prediction probabilities
    proba = model.predict_proba([text])[0]
    
    # Map probabilities to emotions
    emotion_probs = {INV_LABEL_MAPPING[i]: float(p) for i, p in enumerate(proba)}
    
    # Get the most likely emotion
    predicted_emotion = INV_LABEL_MAPPING[proba.argmax()]
    
    return predicted_emotion, emotion_probs


def main():
    """
    Main function to train model or run classification from the command line.
    """
    if len(sys.argv) > 1:
        # If arguments are provided, treat them as input text
        # Join all arguments as they might contain spaces
        input_text = " ".join(sys.argv[1:])
        model = load_model()
        
        # Display detailed prediction
        emotion, probs = predict_with_probabilities(input_text, model)
        
        print(f"\nInput text: \"{input_text}\"")
        print(f"Predicted emotion: {emotion}")
        print("\nProbabilities for each emotion:")
        for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {prob:.4f}")
    else:
        # Interactive mode
        model = load_model()
        
        print("\nEmotion Classifier - Interactive Mode")
        print("Type a sentence to classify its emotion, or 'q' to quit.")
        
        while True:
            # Get input from user
            user_input = input("\nEnter text: ")
            
            # Check if user wants to quit
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            # Skip empty inputs
            if not user_input.strip():
                continue
            
            # Classify and display results
            emotion, probs = predict_with_probabilities(user_input, model)
            
            print(f"\nPredicted emotion: {emotion}")
            print("\nProbabilities for each emotion:")
            for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {prob:.4f}")


if __name__ == "__main__":
    # If run as script, execute main function
    main()