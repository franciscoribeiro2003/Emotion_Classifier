"""Utility functions for the emotion classification project."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud


def plot_class_distribution(df, target_column, figsize=(10, 6), title=None):
    """Plot the class distribution of a target column."""
    plt.figure(figsize=figsize)
    sns.countplot(y=df[target_column], order=df[target_column].value_counts().index)
    plt.title(title or f'Distribution of {target_column}')
    plt.xlabel('Count')
    plt.ylabel(target_column)
    plt.tight_layout()
    return plt.gcf()


def plot_text_length_distribution(df, text_column='text', figsize=(12, 6)):
    """Plot the distribution of text lengths."""
    text_lengths = df[text_column].str.len()
    word_counts = df[text_column].str.split().str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Character length distribution
    sns.histplot(text_lengths, kde=True, ax=axes[0])
    axes[0].set_title('Text Length Distribution (Characters)')
    axes[0].set_xlabel('Number of Characters')
    axes[0].set_ylabel('Frequency')
    
    # Word count distribution
    sns.histplot(word_counts, kde=True, ax=axes[1])
    axes[1].set_title('Text Length Distribution (Words)')
    axes[1].set_xlabel('Number of Words')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig


def create_wordcloud(texts, stop_words=None, max_words=100, figsize=(12, 8)):
    """Create a word cloud from a list of texts."""
    all_text = ' '.join(texts)
    
    wordcloud = WordCloud(
        width=800, 
        height=500, 
        max_words=max_words, 
        stopwords=stop_words,
        background_color='white',
        contour_width=3,
        contour_color='steelblue'
    ).generate(all_text)
    
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    return plt.gcf()


def create_class_wordclouds(df, text_column='text', class_column='sentiment', 
                           stop_words=None, max_words=100, figsize=(15, 10)):
    """Create word clouds for each class in the dataset."""
    classes = df[class_column].unique()
    n_classes = len(classes)
    
    # Determine grid layout
    n_cols = min(3, n_classes)
    n_rows = (n_classes // n_cols) + (1 if n_classes % n_cols != 0 else 0)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_classes > 1 else [axes]
    
    for i, class_name in enumerate(classes):
        class_texts = df[df[class_column] == class_name][text_column]
        all_text = ' '.join(class_texts)
        
        wordcloud = WordCloud(
            width=800, 
            height=500, 
            max_words=max_words, 
            stopwords=stop_words,
            background_color='white',
            contour_width=2,
            contour_color='steelblue'
        ).generate(all_text)
        
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'Class: {class_name}')
        axes[i].axis('off')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    return fig


def get_top_n_words(texts, n=20, ngram_range=(1, 1), stop_words=None):
    """Get the top n most frequent words or n-grams."""
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
    X = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    
    word_counts = np.asarray(X.sum(axis=0)).ravel()
    word_counts_dict = dict(zip(words, word_counts))
    
    sorted_word_counts = sorted(word_counts_dict.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_word_counts[:n]


def plot_top_n_words(texts, n=20, title=None, figsize=(12, 8)):
    """Plot the top n most frequent words."""
    top_words = get_top_n_words(texts, n)
    words, counts = zip(*top_words)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(words)), counts, align='center')
    plt.yticks(range(len(words)), words)
    plt.xlabel('Frequency')
    plt.title(title or f'Top {n} Most Common Words')
    plt.tight_layout()
    
    return plt.gcf()
