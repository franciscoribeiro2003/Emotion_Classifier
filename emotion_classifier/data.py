"""Functions for loading and processing the EmotionVerse dataset."""

import pandas as pd
from datasets import load_dataset


from emotion_classifier.config import (
    DATASET_NAME,
    DATA_DIR,
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
