# from emotion_classifier.preprocessing import preprocess_text, clean_text, remove_stopword
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def clean_text(text, lower=True, remove_punctuation=True, remove_emoji=True, remove_digits=True):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase if specified
    if lower:
        text = text.lower()
    
    # Remove punctuation if specified
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove Emojis
    if remove_emoji:
        text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove digits if specified
    if remove_digits:
        text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """Tokenize text into words."""
    return word_tokenize(text)


def remove_stopwords(tokens, language='english'):
    """Remove stopwords from list of tokens."""
    stop_words = set(stopwords.words(language))
    return [token for token in tokens if token not in stop_words]


def stem_tokens(tokens):
    """Stem tokens using Porter stemmer."""
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens):
    """Lemmatize tokens using WordNet lemmatizer."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(
    text, 
    lower=True, 
    remove_punct=True,
    remove_digits=True,
    remove_stops=True,
    stemming=False,
    lemmatization=True,
    language='english'
):
    """Full text preprocessing pipeline."""
    # Clean text
    text = clean_text(text, lower, remove_punct, remove_digits)
    
    # Tokenize
    tokens = tokenize_text(text)
    
    # Remove stopwords if specified
    if remove_stops:
        tokens = remove_stopwords(tokens, language)
    
    # Apply stemming if specified
    if stemming:
        tokens = stem_tokens(tokens)
    
    # Apply lemmatization if specified
    if lemmatization:
        tokens = lemmatize_tokens(tokens)
    
    return tokens

def preprocess_and_replace(text):
    return ' '.join(preprocess_text(
        text, 
        lower=True, 
        remove_punct=True, 
        remove_digits=True,
        remove_stops=True,
        stemming=True,  # Apply stemming
        lemmatization=False  # Ensure lemmatization is off
    ))