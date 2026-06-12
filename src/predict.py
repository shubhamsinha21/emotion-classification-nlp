import joblib

from preprocess import preprocess_text


MODEL_PATH = "models/emotion_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


model = joblib.load(
    MODEL_PATH
)

vectorizer = joblib.load(
    VECTORIZER_PATH
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

    return {
        "input_text": text,
        "processed_text": processed_text,
        "predicted_emotion": prediction,
        "model": "Linear SVC"
    }


if __name__ == "__main__":

    sample_text = (
        "I am feeling very happy today"
    )

    result = predict_emotion(
        sample_text
    )

    print("\nPrediction Result:\n")

    print(result)