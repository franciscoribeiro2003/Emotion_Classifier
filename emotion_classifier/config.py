"""Configuration settings for the emotion classification project."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"


# Dataset
DATASET_NAME = "ayjays132/Emotionverse"
