# Emotion_Classifier
M.EIC022 FEUP Processamento de Linguagem Natural PLN assignment project

## Overview

### Assigment 1
This project focuses on building NLP classifiers for emotion classification using the EmotionVerse dataset [ayjays132/Emotionverse](https://huggingface.co/datasets/ayjays132/Emotionverse). It explores various preprocessing techniques, feature extraction methods, and traditional machine learning classifiers to understand and predict emotional content in text.

To analyse our aproach, please check out the three notebooks:
- notebooks/**01-exploratory-data-analysis**.ipynb: Initial data exploration and visualization
- notebooks/**02-preprocessing-and-feature-extraction**.ipynb: Text preprocessing and feature engineering
- notebooks/**03-model-training-and-evaluation**.ipynb: Model training, comparison and error analysis

## Dataset

The EmotionVerse dataset is a comprehensive collection of emotional texts designed for AI emotion understanding:

- **Sentiment**: 5 categories (Mixed, Positive, Negative, Ambiguous, Neutral)
- **Primary Emotions**: 94 unique emotional categories
- **Secondary Emotions**: 126 unique emotions
- **Meta Emotions**: 1659 diverse emotional expressions
- **Context**: 164 unique situations (Relationships, Career, Self-Reflection, etc.)
- **Interaction Style**: 27 types of user interaction

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


## NRC Emotion Lexicon

The project uses the [NRC Word-Emotion Association Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) (EmoLex) created by Saif M. Mohammad and Peter D. Turney. This lexicon maps words to eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive), providing valuable domain knowledge for emotion classification.

Citation:
- Saif Mohammad and Peter Turney. "Emotions Evoked by Common Words and Phrases: Using Mechanical Turk to Create an Emotion Lexicon." In Proceedings of the NAACL-HLT Workshop on Computational Approaches to Analysis and Generation of Emotion in Text, June 2010.


## Getting Started

### Installation
1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`

### Running or analyse the results
Run/See the notebooks in order (01 → 02 → 03) to reproduce the analysis and results

### Test Text-Only Emotion Classifier - Fun Feature

For demonstration purposes, we've included a simple command-line tool that allows you to test emotion classification on your own text:

```bash
python emotion_classifier/emotion_classifier.py <text: optional>
```
If text not provided, then the script falls back to a interactive mode.

**Note:** This text-only classifier is provided for interactive experimentation and doesn't represent the best performing approach from our comprehensive evaluation in Assignment 1.

**Interaction Example**:

```
Emotion Classifier - Interactive Mode
Type a sentence to classify its emotion, or 'q' to quit.

Enter text: "Ever since I met you, everything I ever cared about is gone! Ruined, turned to [beep], dead, ever since I hooked up with the great Heisenberg! I have never been more alone! I HAVE NOTHING! NO ONE! ALRIGHT, IT'S ALL GONE, GET IT?"

Predicted emotion: Negative

Probabilities for each emotion:
  Negative: 0.9967
  Neutral: 0.0017
  Positive: 0.0008
  Ambiguous: 0.0005
  Mixed: 0.0003
```

The script trains on first use and saves the model for faster subsequent runs.