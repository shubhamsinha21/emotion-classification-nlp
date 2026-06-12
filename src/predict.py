import joblib

from preprocess import preprocess_text


# Load model

model = joblib.load(
    "models/emotion_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


def predict_emotion(text):

    processed_text = preprocess_text(
        text
    )

    vectorized_text = vectorizer.transform(
        [processed_text]
    )

    prediction = model.predict(
        vectorized_text
    )[0]

    return prediction


if __name__ == "__main__":

    sample_text = (
        "I am feeling very happy today"
    )

    emotion = predict_emotion(
        sample_text
    )

    print("\nInput:")
    print(sample_text)

    print("\nPredicted Emotion:")
    print(emotion)