import re
import nltk

from nltk.corpus import stopwords


def download_nltk_resources():
    """
    Download required NLTK resources.
    Streamlit Cloud safe version.
    """

    try:
        nltk.data.find("corpora/stopwords")

    except LookupError:
        nltk.download("stopwords")


download_nltk_resources()

STOP_WORDS = set(
    stopwords.words("english")
)


def clean_text(text):
    """
    Basic text cleaning
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
    Remove stop words
    """

    tokens = text.split()

    filtered_tokens = [
        word
        for word in tokens
        if word not in STOP_WORDS
    ]

    return " ".join(
        filtered_tokens
    )


def preprocess_text(text):
    """
    Complete preprocessing pipeline
    """

    text = clean_text(text)

    text = remove_stopwords(text)

    return text