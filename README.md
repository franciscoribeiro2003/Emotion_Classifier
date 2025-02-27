# Emotion_Classifier
M.EIC022 FEUP Processamento de Linguagem Natural PLN assignment project

## Overview

This project focuses on building NLP classifiers for emotion classification using the EmotionVerse dataset [ayjays132/Emotionverse](https://huggingface.co/datasets/ayjays132/Emotionverse). It explores various preprocessing techniques, feature extraction methods, and traditional machine learning classifiers to understand and predict emotional content in text.

## Dataset

The EmotionVerse dataset is a comprehensive collection of emotional texts designed for AI emotion understanding:

- **Sentiment**: 5 categories (Mixed, Positive, Negative, Ambiguous, Neutral)
- **Primary Emotions**: 94 unique emotional categories
- **Secondary Emotions**: 126 unique emotions
- **Meta Emotions**: 1659 diverse emotional expressions
- **Context**: 164 unique situations (Relationships, Career, Self-Reflection, etc.)
- **Interaction Style**: 27 types of user interaction

### Data Distribution
- Mixed: 652 samples
- Positive: 542 samples
- Negative: 419 samples
- Ambiguous: 158 samples
- Neutral: 103 samples

## Project Structure

- `notebooks/`: Jupyter notebooks for exploratory analysis and model development
- `emotion_classifier/`: Main package with all the code
- `data/`: Data storage
- `scripts/`: Utility scripts for downloading data, etc
- `docs/`: Report and documentation
