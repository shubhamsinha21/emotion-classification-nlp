import pandas as pd

from preprocess import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.svm import LinearSVC
import joblib


def load_dataset(file_path):
    """
    Load emotion dataset
    """

    df = pd.read_csv(
        file_path,
        sep=";",
        names=["text", "emotion"]
    )

    return df


if __name__ == "__main__":

    # Load dataset
    df = load_dataset("data/train.txt")

    # Missing values
    print("\nMissing Values:\n")
    print(df.isnull().sum())

    # Preprocess text
    df["processed_text"] = df["text"].apply(
        preprocess_text
    )

    print("\nDataset Shape:")
    print(df.shape)

    # Features and Labels
    X = df["processed_text"]
    y = df["emotion"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    print("\nTF-IDF Shapes:")
    print("Train:", X_train_tfidf.shape)
    print("Test :", X_test_tfidf.shape)

    print("\nVocabulary Size:")
    print(len(vectorizer.vocabulary_))
    
    # Logistic Regression Model

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    lr_model.fit(
        X_train_tfidf,
        y_train
    )

    lr_predictions = lr_model.predict(
        X_test_tfidf
    )

    lr_accuracy = accuracy_score(
        y_test,
        lr_predictions
    )

    print(f"\nAccuracy: {lr_accuracy:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            lr_predictions
        )
    )

    print("\n" + "=" * 60)
    print("LINEAR SVC")
    print("=" * 60)

    svm_model = LinearSVC(
        random_state=42
    )

    svm_model.fit(
        X_train_tfidf,
        y_train
    )

    svm_predictions = svm_model.predict(
        X_test_tfidf
    )

    svm_accuracy = accuracy_score(
        y_test,
        svm_predictions
    )

    print(f"\nAccuracy: {svm_accuracy:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            svm_predictions
        )
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(f"Logistic Regression : {lr_accuracy:.4f}")
    print(f"Linear SVC          : {svm_accuracy:.4f}")

    if svm_accuracy > lr_accuracy:
        print("\nWinner: Linear SVC")
    else:
        print("\nWinner: Logistic Regression")
        
        
print(f"Logistic Regression : {lr_accuracy:.4f}")
print(f"Linear SVC          : {svm_accuracy:.4f}")

# Save best model

joblib.dump(
        svm_model,
        "models/emotion_model.pkl"
    )

joblib.dump(
        vectorizer,
        "models/tfidf_vectorizer.pkl"
    )

print("\nModel Saved Successfully!")