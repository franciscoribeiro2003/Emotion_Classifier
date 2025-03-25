"""Lexicon and Categorical feature transformation utilities for emotion classification."""

import os
import pickle
import requests
import numpy as np
import numpy as np
import ast

from emotion_classifier.config import (
    NRC_LEXICON_URL,
    NRC_PATH
)
from emotion_classifier.preprocessing import preprocess_text


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


def create_sentiment_probability_mapping(df, categorical_col, sentiment_col='sentiment'):
    """
    Create a probability distribution mapping from a categorical column to sentiment.
    
    Args:
        df (DataFrame): The DataFrame containing the data
        categorical_col (str): Name of the categorical column to map
        sentiment_col (str): Name of the sentiment column (default: 'sentiment')
        
    Returns:
        DataFrame: Normalized probability distribution showing relationship between
                  the categorical column values and sentiment classes
    """
    # Group by categorical column and sentiment to count occurrences
    counts = df.groupby([categorical_col, sentiment_col]).size().reset_index(name='count')
    
    # Pivot the data
    pivot = counts.pivot_table(index=categorical_col, 
                               columns=sentiment_col, 
                               values='count', 
                               fill_value=0)
    
    # Normalize the pivot table by rows
    normalized = pivot.div(pivot.sum(axis=1), axis=0)
    
    return normalized


def map_categorical_to_numeric(df, column, probability_map, sentiment_mapping, default_value=0.5):
    """
    Transform categorical values to numeric based on their probabilistic relationship with sentiment.
    
    Args:
        df (DataFrame or Series): Data containing the categorical column
        column (str): Name of the categorical column to transform
        probability_map (DataFrame): Probability distribution from create_sentiment_probability_mapping
        sentiment_mapping (dict): Mapping from sentiment categories to numeric values
        default_value (float): Default value for categories not in the probability map
        
    Returns:
        list: Numerical values corresponding to the categorical values
    """
    result = []
    
    for value in df[column]:
        if value in probability_map.index:
            # Get probability distribution for this categorical value
            probabilities = probability_map.loc[value].values
            sentiment_categories = probability_map.columns
            
            # Compute cumulative probability intervals
            cumulative_probs = np.cumsum(probabilities)
            random_value = np.random.rand()
            
            # Find the corresponding sentiment based on probability distribution
            selected_sentiment = sentiment_categories[np.searchsorted(cumulative_probs, random_value)]
            result.append(sentiment_mapping[selected_sentiment])
        else:
            # Default to neutral if value is not in our probability table
            result.append(default_value)
    
    return result


def transform_primary_emotion(df, probability_map, sentiment_mapping):
    """
    Transform primary_emotion categorical values to numeric values.
    
    Args:
        df (DataFrame): DataFrame containing primary_emotion column
        probability_map (DataFrame): Probability distribution from create_sentiment_probability_mapping
        sentiment_mapping (dict): Mapping from sentiment categories to numeric values
        
    Returns:
        list: Numeric values for the primary_emotion column
    """
    return map_categorical_to_numeric(df, 'primary_emotion', probability_map, sentiment_mapping)


def transform_interaction_style(df, probability_map, sentiment_mapping):
    """
    Transform interaction_style categorical values to numeric values.
    
    Args:
        df (DataFrame): DataFrame containing interaction_style column
        probability_map (DataFrame): Probability distribution from create_sentiment_probability_mapping
        sentiment_mapping (dict): Mapping from sentiment categories to numeric values
        
    Returns:
        list: Numeric values for the interaction_style column
    """
    return map_categorical_to_numeric(df, 'interaction_style', probability_map, sentiment_mapping)


def transform_context(df, probability_map, sentiment_mapping):
    """
    Transform context categorical values to numeric values.
    
    Args:
        df (DataFrame): DataFrame containing context column
        probability_map (DataFrame): Probability distribution from create_sentiment_probability_mapping
        sentiment_mapping (dict): Mapping from sentiment categories to numeric values
        
    Returns:
        list: Numeric values for the context column
    """
    return map_categorical_to_numeric(df, 'context', probability_map, sentiment_mapping)


def transform_secondary_emotions(df, probability_map, sentiment_mapping):
    """
    Transform secondary_emotions lists to numeric values based on their 
    probabilistic relationship with sentiment.

    This function is more developed because the secondary_emotions column contains pairs of emotions,
    so it is needed to calculate the average probability of each pair and then
    select a sentiment based on that probability distribution.

    
    Args:
        df (DataFrame): DataFrame containing secondary_emotions column
        probability_map (DataFrame): Probability distribution for individual emotions
        sentiment_mapping (dict): Mapping from sentiment categories to numeric values
        
    Returns:
        list: Numeric values representing the secondary_emotions
    """
    result = []
    
    for emotions in df['secondary_emotions']:
        # Handle different formats of secondary_emotions
        if isinstance(emotions, list) and len(emotions) == 2:
            emotion_str = [str(e) for e in emotions]
        else:
            emotions_str = str(emotions).strip("[]").split(', ')
            emotion_str = [e.strip("'") for e in emotions_str]
        
        if len(emotion_str) == 2:
            # Get probabilities for each emotion
            probs1 = probability_map.loc[emotion_str[0]].values if emotion_str[0] in probability_map.index else np.array([0.2]*5)
            probs2 = probability_map.loc[emotion_str[1]].values if emotion_str[1] in probability_map.index else np.array([0.2]*5)
            
            # Average the probabilities
            avg_probs = (probs1 + probs2) / 2
            sentiment_categories = probability_map.columns
            
            # Select sentiment based on probability distribution
            cumulative_probs = np.cumsum(avg_probs)
            random_value = np.random.rand()
            
            selected_sentiment = sentiment_categories[np.searchsorted(cumulative_probs, random_value)]
            result.append(sentiment_mapping[selected_sentiment])
        else:
            result.append(0.5)  # Default to neutral if pair isn't recognized
    
    return result


def prepare_secondary_emotions_mapping(train_df):
    """
    Prepare probability mapping for secondary emotions.
    
    Args:
        train_df (DataFrame): Training DataFrame with secondary_emotions and sentiment columns
        
    Returns:
        DataFrame: Normalized probability distribution for secondary emotions
    """
    # Ensure secondary_emotions are properly formatted
    train_df = train_df.copy()
    train_df['secondary_emotions'] = train_df['secondary_emotions'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Explode secondary_emotions for analysis
    train_exploded = train_df.assign(
        secondary_emotion=train_df['secondary_emotions'].astype(str).str.strip("[]").str.split(', ')
    )
    train_exploded = train_exploded.explode('secondary_emotion')
    
    # Clean up the secondary_emotion column
    train_exploded['secondary_emotion'] = train_exploded['secondary_emotion'].str.strip("'")
    
    # Create the probability mapping
    sec_emotion_counts = train_exploded.groupby(['secondary_emotion', 'sentiment']).size().reset_index(name='count')
    sec_emotion_pivot = sec_emotion_counts.pivot_table(
        index='secondary_emotion', 
        columns='sentiment', 
        values='count', 
        fill_value=0
    )
    sec_emotion_normalized = sec_emotion_pivot.div(sec_emotion_pivot.sum(axis=1), axis=0)
    
    return sec_emotion_normalized

def transform_meta_emotions(df, probability_map, sentiment_mapping):
    """
    Transform meta_emotions tuple to numeric values based on their 
    probabilistic relationship with sentiment.

    This function is more developed because the meta_emotions column consist in a string of 
    singles or pairs of descriptions of emotions, so it is needed to calculate the average probability
    of each group of emotions and then select a sentiment based on that probability distribution.
    
    Args:
        df (DataFrame): DataFrame containing meta_emotions column
        probability_map (DataFrame): Probability distribution for individual emotions
        sentiment_mapping (dict): Mapping from sentiment categories to numeric values
        
    Returns:
        list: Numeric values representing the meta_emotions
    """
    result = []
    
    for emotions in df['meta_emotions']:
        if isinstance(emotions, tuple):
            probs1 = probability_map.loc[emotions].values if emotions in probability_map.index else np.array([0.2]*5)
            avg_probs = probs1  # Average probability
            sentiment_categories = probability_map.columns
            
            cumulative_probs = np.cumsum(avg_probs)
            random_value = np.random.rand()
            
            selected_sentiment = sentiment_categories[np.searchsorted(cumulative_probs, random_value)]
            result.append(sentiment_mapping[selected_sentiment])
        else:
            result.append(0.5)  # Default to neutral if group isn't recognized
    
    return result


def prepare_meta_emotions_mapping(train_df):
    """
    Prepare probability mapping for meta emotions.
    
    Args:
        train_df (DataFrame): Training DataFrame with meta_emotions and sentiment columns
    
    Returns:
        DataFrame: Normalized probability distribution for meta emotions
    """
    # Process meta_emotion column from strings to a list of preprocessed descriptions of emotions
    train_df['meta_emotions'] = train_df['meta_emotions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    train_df['meta_emotions'] = train_df['meta_emotions'].apply(lambda x: [preprocess_text(emotion) for emotion in x] if isinstance(x, list) else x)

    # Flatten the list of lists and sort the emotions into tuple to simplify comparison
    train_df['meta_emotions'] = train_df['meta_emotions'].apply(lambda x: [item for sublist in x for item in (sublist if isinstance(sublist, list) else [sublist])])
    train_df['meta_emotions'] = train_df['meta_emotions'].apply(lambda x: tuple(sorted(x)) if isinstance(x, list) else x)

    # Explode groups into individual emotions for probability analysis
    train_df_exploded = train_df.assign(meta_emotion=train_df['meta_emotions'])
    train_df_exploded = train_df_exploded.explode('meta_emotions')

    # Compute sentiment probability distribution for individual meta emotions
    meta_emotion_sentiment_counts = train_df_exploded.groupby(['meta_emotions', 'sentiment']).size().reset_index(name='count')
    meta_emotion_sentiment_pivot = meta_emotion_sentiment_counts.pivot_table(index='meta_emotions', 
                                                                            columns='sentiment', 
                                                                            values='count', 
                                                                            fill_value=0)
    meta_emotion_sentiment_normalized = meta_emotion_sentiment_pivot.div(meta_emotion_sentiment_pivot.sum(axis=1), axis=0)

    return meta_emotion_sentiment_normalized


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

