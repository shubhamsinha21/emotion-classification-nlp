import re
import nltk
from nltk.corpus import stopwords


def download_nltk_resources():
    """
    Download stopwords if not available.
    """

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


download_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    """
    Clean text.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def remove_stopwords(text):
    """
    Remove stopwords using split().
    No word_tokenize().
    """

    tokens = text.split()

    filtered_tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

    return " ".join(filtered_tokens)


def preprocess_text(text):
    """
    Complete preprocessing pipeline.
    """

    text = clean_text(text)

    text = remove_stopwords(text)

    return text