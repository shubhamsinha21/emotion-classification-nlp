import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def download_nltk_resources():
    """
    Download required NLTK resources.
    Works both locally and on Streamlit Cloud.
    """

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    for resource_path, package_name in resources:

        try:
            nltk.data.find(resource_path)

        except LookupError:
            nltk.download(package_name)


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