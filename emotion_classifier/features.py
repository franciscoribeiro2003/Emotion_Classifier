"""Lexicon and Categorical feature transformation utilities for emotion classification."""

import os
import pickle
import requests
import numpy as np
import numpy as np
import ast
from sklearn.preprocessing import OneHotEncoder

from emotion_classifier.config import (
    NRC_LEXICON_URL,
    NRC_PATH
)


def load_nrc_emotion_lexicon():
    """
    Download and load the full NRC Emotion Lexicon.
    
    The NRC Emotion Lexicon is a list of English words and their associations with
    eight basic emotions (anger, fear, anticipation, trust, surprise, sadness,
    joy, and disgust) and two sentiments (negative and positive).
    
    Returns:
        dict: A dictionary mapping emotion categories to lists of words
    """
    
    # Check if we've already downloaded and processed the lexicon
    if os.path.exists(NRC_PATH):
        print("Loading NRC Emotion Lexicon from disk...")
        with open(NRC_PATH, 'rb') as f:
            return pickle.load(f)
    
    print("Downloading NRC Emotion Lexicon...")
    
    response = requests.get(NRC_LEXICON_URL)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Process the lexicon file
    lines = response.text.split('\n')
    
    # Initialize the lexicon dictionary
    lexicon = {
        'anger': set(),
        'anticipation': set(),
        'disgust': set(), 
        'fear': set(),
        'joy': set(),
        'negative': set(),
        'positive': set(),
        'sadness': set(),
        'surprise': set(),
        'trust': set()
    }
    
    # Parse the lexicon file
    for line in lines:
        if not line or line.startswith('#'):
            continue
            
        parts = line.strip().split('\t')
        if len(parts) == 3:
            word, emotion, flag = parts
            if int(flag) == 1:  # If the word has this emotion
                if emotion in lexicon:
                    lexicon[emotion].add(word)
    
    # Convert sets to lists
    for emotion in lexicon:
        lexicon[emotion] = list(lexicon[emotion])
        
    print(f"Processed NRC Emotion Lexicon with {sum(len(words) for words in lexicon.values())} word-emotion associations")
    
    # Save the lexicon to disk
    os.makedirs(os.path.dirname(NRC_PATH), exist_ok=True)
    with open(NRC_PATH, 'wb') as f:
        pickle.dump(lexicon, f)
        
    return lexicon


def extract_lexicon_features(texts, lexicons):
    """
    Extract emotion lexicon features from texts.
    
    For each text, count the occurrences of words associated with each emotion
    in the lexicon and create a feature vector.
    
    Args:
        texts (list): List of preprocessed text strings
        lexicons (dict): Dictionary mapping emotion categories to lists of words
        
    Returns:
        numpy.ndarray: Array of shape (len(texts), len(lexicons)) containing lexicon features
    """
    features = np.zeros((len(texts), len(lexicons)))
    
    for i, text in enumerate(texts):
        words = set(text.split())
        for j, (emotion, emotion_words) in enumerate(lexicons.items()):
            # Count matches between text and emotion lexicon
            matches = sum(1 for word in words if word in emotion_words)
            features[i, j] = matches
            
    return features


def combine_categorical_features(df, columns_to_combine):
    """
    Combine categorical text columns into a single text field.
    
    Args:
        df (DataFrame): DataFrame containing categorical columns
        columns_to_combine (list): List of column names to combine
        
    Returns:
        list: Combined text strings for each row
    """
    combined_texts = []
    
    for _, row in df.iterrows():
        combined_parts = []
        
        for col in columns_to_combine:
            if col in row and row[col] is not None:
                # Handle different data types appropriately
                if isinstance(row[col], list):
                    # Join list items with spaces
                    combined_parts.append(' '.join(str(item) for item in row[col]))
                else:
                    # Just add the string value
                    combined_parts.append(str(row[col]))
        
        # Join all parts with spaces
        combined_texts.append(' '.join(combined_parts))
    
    return combined_texts


def normalize_intensity(df, max_value=10):
    """
    Normalize the intensity column to a [0-1] range.
    
    Args:
        df (DataFrame): DataFrame containing the intensity column
        max_value (int): Maximum intensity value for normalization
        
    Returns:
        numpy.ndarray: Normalized intensity values
    """
    return df['intensity'] / max_value


def create_one_hot_features(df, categorical_columns):
    """
    Create one-hot encoded features for categorical columns.
    
    Args:
        df (DataFrame): DataFrame containing categorical columns
        categorical_columns (list): List of column names to encode
        
    Returns:
        tuple: (encoded_features, encoder) where encoded_features is a sparse matrix
               and encoder is the fitted OneHotEncoder
    """
    # Handle list columns (like secondary_emotions)
    df_processed = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            # Convert lists to strings for one-hot encoding
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df_processed[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    # Create and fit the encoder
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df_processed[categorical_columns])
    
    return encoded_features, encoder


def apply_one_hot_encoding(df, encoder, categorical_columns):
    """
    Apply a fitted one-hot encoder to new data.
    
    Args:
        df (DataFrame): DataFrame containing categorical columns
        encoder (OneHotEncoder): Fitted OneHotEncoder
        categorical_columns (list): List of column names to encode
        
    Returns:
        sparse matrix: One-hot encoded features
    """
    # Handle list columns (like secondary_emotions)
    df_processed = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            # Convert lists to strings for one-hot encoding
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df_processed[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    return encoder.transform(df_processed[categorical_columns])

def encode_secondary_emotions(df, all_emotions):
    """Create a multi-hot encoding for secondary emotions."""
    # Initialize matrix with zeros
    encoded = np.zeros((len(df), len(all_emotions)))
    
    # Map emotions to indices
    emotion_to_idx = {emotion: i for i, emotion in enumerate(all_emotions)}
    
    # Fill in the matrix
    for i, emotions in enumerate(df['secondary_emotions']):
        if isinstance(emotions, list):
            for emotion in emotions:
                if emotion in emotion_to_idx:
                    encoded[i, emotion_to_idx[emotion]] = 1
        elif isinstance(emotions, str) and emotions.startswith('['):
            # Handle string representations of lists
            try:
                emotion_list = ast.literal_eval(emotions)
                for emotion in emotion_list:
                    if emotion in emotion_to_idx:
                        encoded[i, emotion_to_idx[emotion]] = 1
            except:
                pass
                
    return encoded