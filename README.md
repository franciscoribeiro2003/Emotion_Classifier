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
  - `01-exploratory-data-analysis.ipynb`: Initial data exploration and visualization
  - `02-preprocessing-and-feature-extraction.ipynb`: Text preprocessing and feature engineering
  - `03-model-training-and-evaluation.ipynb`: Model training, comparison and error analysis
- `emotion_classifier/`: Main package with all the code
  - `data.py`: Functions for loading and processing dataset
  - `preprocessing.py`: Text preprocessing utilities
  - `features.py`: Feature extraction functions including lexicon processing
  - `categorical.py`: Categorical feature transformation utilities
- `data/`: Data storage
- `docs/`: Report and documentation

## Feature Extraction

The project implements multiple feature representations:

### Text-Based Features

| Feature Type | Description | Dimensions | Density |
|-------------|-------------|------------|---------|
| Bag-of-Words | Word frequency counts | (1124, 1447) | 0.89% |
| TF-IDF | Term frequency-inverse document frequency | (1124, 1447) | 0.89% |
| Word2Vec | Custom trained word embeddings | (1124, 100) | Dense |
| NRC Lexicon | Emotion scores from lexicon | (1124, 10) | Dense |

### Categorical Features

| Feature | Description |
|---------|-------------|
| Primary Emotion | Transformed to 0-1 scale based on sentiment probability |
| Secondary Emotions | Transformed to 0-1 scale based on sentiment probability |
| Interaction Style | Transformed to 0-1 scale based on sentiment probability |
| Context | Transformed to 0-1 scale based on sentiment probability |
| Intensity | Normalized from 1-10 scale to 0-1 |

### Combined Feature Sets

Various combinations of the above features were created to evaluate which representation best captures emotional content.

## NRC Emotion Lexicon

The project uses the [NRC Word-Emotion Association Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) (EmoLex) created by Saif M. Mohammad and Peter D. Turney. This lexicon maps words to eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive), providing valuable domain knowledge for emotion classification.

Citation:
- Saif Mohammad and Peter Turney. "Emotions Evoked by Common Words and Phrases: Using Mechanical Turk to Create an Emotion Lexicon." In Proceedings of the NAACL-HLT Workshop on Computational Approaches to Analysis and Generation of Emotion in Text, June 2010.

## Results

Multiple classification models were evaluated across different feature representations:

| Model Type | Best Feature Set | Validation Accuracy | Test Accuracy | Macro F1 |
|------------|------------------|---------------------|---------------|----------|
| XGBoost | All Features Combined | 78.13% | 76.27% | 0.7262 |
| MLP | All Features Combined | 77.60% | - | 0.7405 |
| XGBoost | TF-IDF + Categorical | 76.00% | - | 0.7343 |
| MLP | TF-IDF + Categorical | 75.47% | - | 0.7161 |
| SVM | TF-IDF | 74.13% | - | 0.6837 |

The best overall model was XGBoost using all features combined, achieving 76.27% accuracy and 0.7262 F1 score on the test set, significantly outperforming the majority class baseline (34.93% accuracy).

## Getting Started

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the notebooks in order (01 → 02 → 03) to reproduce the analysis and results