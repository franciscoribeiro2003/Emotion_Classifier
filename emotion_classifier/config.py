"""Configuration settings for the emotion classification project."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"


# Dataset
DATASET_NAME = "ayjays132/Emotionverse"
# Processed data
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Data splits
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25

# Lexicon
NRC_LEXICON_URL = "https://archive.org/download/nrc-emotion-lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
NRC_LOCATION = DATA_DIR / "lexicons"
NRC_PATH = NRC_LOCATION / "nrc_lexicon.pkl"