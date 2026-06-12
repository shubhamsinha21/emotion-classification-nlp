import streamlit as st
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(
    str(
        Path(__file__).resolve().parent.parent / "src"
    )
)

from predict import predict_emotion

st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="😊",
    layout="wide"
)

# -------------------------------
# Emotion Mapping
# -------------------------------

EMOTION_EMOJIS = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# -------------------------------
# Header
# -------------------------------

st.title("😊 Emotion Detection System")

st.markdown(
    """
    Detect emotions from text using NLP and Machine Learning.

    **Model:** Linear SVC  
    **Accuracy:** 88.81%
    """
)

st.divider()

# -------------------------------
# Main Layout
# -------------------------------

col1, col2 = st.columns([2, 1])

with col1:

    user_text = st.text_area(
        "Enter Text",
        placeholder="Type your text here...",
        height=200
    )

    if st.button("Predict Emotion"):

        if user_text.strip():

            result = predict_emotion(
                user_text
            )

            emotion = result[
                "predicted_emotion"
            ]

            emoji = EMOTION_EMOJIS.get(
                emotion,
                "😊"
            )

            st.success(
                f"{emoji} Predicted Emotion: {emotion.upper()}"
            )

            with st.expander(
                "Prediction Details"
            ):
                st.write(
                    f"**Original Text:** {result['input_text']}"
                )

                st.write(
                    f"**Processed Text:** {result['processed_text']}"
                )

                st.write(
                    f"**Model Used:** {result['model']}"
                )

        else:
            st.warning(
                "Please enter some text."
            )

with col2:

    st.subheader("📊 Project Metrics")

    st.metric(
        "Accuracy",
        "88.81%"
    )

    st.metric(
        "Training Samples",
        "12,800"
    )

    st.metric(
        "Testing Samples",
        "3,200"
    )

    st.metric(
        "Vocabulary Size",
        "5,000"
    )

# -------------------------------
# Confusion Matrix
# -------------------------------

st.divider()

st.subheader(
    "Model Evaluation"
)

confusion_matrix_path = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "confusion_matrix.png"
)

st.image(
    str(confusion_matrix_path),
    caption="Confusion Matrix"
)

# -------------------------------
# Footer
# -------------------------------

st.divider()

st.caption(
    "Built using TF-IDF + Linear SVC for Emotion Classification"
)