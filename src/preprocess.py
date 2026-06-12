import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def download_nltk_resources():

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    try:
        word_tokenize("test")
    except LookupError:
        nltk.download("punkt")


download_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    """
    Basic text cleaning
    """

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text):
    """
    Remove stop words
    """

    tokens = word_tokenize(text)

    filtered_tokens = [
        word
        for word in tokens
        if word not in STOP_WORDS
    ]

    return " ".join(filtered_tokens)


def preprocess_text(text):
    """
    Complete preprocessing pipeline
    """

    text = clean_text(text)

    text = remove_stopwords(text)

    return text