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

# Example Inputs

    st.subheader("Try Example Inputs")

    example_col1, example_col2, example_col3 = st.columns(3)

    with example_col1:

        if st.button("😄 Joy Example"):
            st.session_state.example_text = (
             "I am feeling very happy today"
        )

        if st.button("😢 Sad Example"):
            st.session_state.example_text = (
            "I miss my old friends and feel lonely"
        )

    with example_col2:

        if st.button("😨 Fear Example"):
            st.session_state.example_text = (
                "I am extremely nervous about tomorrow"
            )

        if st.button("😡 Anger Example"):
            st.session_state.example_text = (
                "I am furious about what happened"
            )

    with example_col3:

        if st.button("❤️ Love Example"):
            st.session_state.example_text = (
                "I really love spending time with my family"
            )

        if st.button("😲 Surprise Example"):
            st.session_state.example_text = (
                "I cannot believe this happened to me"
            )

    # Default value

    if "example_text" not in st.session_state:
        st.session_state.example_text = ""

    user_text = st.text_area(
        "Enter Text",
        value=st.session_state.example_text,
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
    
    st.divider()

    st.subheader("🚀 Project Highlights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
            **Dataset**
            
            16,000 Samples
            
            6 Emotion Classes
            """
        )

    with col2:
        st.info(
            """
            **Best Model**
            
            Linear SVC
            
            88.81% Accuracy
            """
        )

    with col3:
        st.info(
            """
            **Features**
            
            TF-IDF
            
            5,000 Features
            """
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