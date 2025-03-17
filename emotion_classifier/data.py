"""Functions for loading and processing the EmotionVerse dataset."""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


from emotion_classifier.config import (
    DATASET_NAME,
    DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
)


def download_dataset():
    """Download the EmotionVerse dataset from Hugging Face."""
    dataset = load_dataset(DATASET_NAME)
    
    # Convert to DataFrame and save to CSV
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"]) if "test" in dataset else None
    
    # Save raw data
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    if test_df is not None:
        test_df.to_csv(DATA_DIR / "test.csv", index=False)
    
    return train_df, test_df

def load_raw_data():
    """Load the raw EmotionVerse dataset from disk."""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    
    train_df = pd.read_csv(train_path) if train_path.exists() else None
    test_df = pd.read_csv(test_path) if test_path.exists() else None
    
    if train_df is None:
        train_df, test_df = download_dataset()
    
    return train_df, test_df


def get_dataset_stats(df, target_column="sentiment"):
    """Get basic statistics about the dataset."""
    stats = {
        "Total samples": len(df),
        "Class distribution": df[target_column].value_counts().to_dict(),
        "Class distribution (%)": (df[target_column].value_counts(normalize=True) * 100).round(2).to_dict(),
        "Average text length (characters)": df["text"].str.len().mean(),
        "Average text length (words)": df["text"].str.split().str.len().mean(),
        "Number of unique emotions": len(df["primary_emotion"].unique())
    }
    
    return stats


def prepare_dataset_splits(df, target_column="sentiment"):
    """
    Prepare train, validation, and test splits.
    If test_df is provided, use it as the test set.
    Otherwise, split the train_df into train, validation, and test.
    """
    # First, split into train+val and test
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df[target_column]
    )
    
    # Then split train+val into train and val
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=VAL_SIZE, 
        random_state=RANDOM_SEED,
        stratify=train_val_df[target_column]
    )
    
    # Save processed splits
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    
    return train_df, val_df, test_df

