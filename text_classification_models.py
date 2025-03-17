import streamlit as st
import pandas as pd
import re
import numpy as np  # Added for np.max()
import joblib  # or pickle/tensorflow/keras depending on your model
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_models():
    models = {}
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pkl"):
            model_name = os.path.splitext(file)[0]
            model_path = os.path.join(MODEL_DIR, file)
            models[model_name] = joblib.load(model_path)
    return models

TFIDF_DIR = "tfidf"
os.makedirs(TFIDF_DIR, exist_ok=True)

def load_tfidf_vectorizer(): 
    tfidf_vectorizers = {}
    for file in os.listdir(TFIDF_DIR):
        if file.endswith(".pkl"):
            vectorizer_name = os.path.splitext(file)[0]
            vectorizer_path = os.path.join(TFIDF_DIR, file)
            tfidf_vectorizers[vectorizer_name] = joblib.load(vectorizer_path)
    return tfidf_vectorizers

st.sidebar.title("Model Selection")
available_models = load_models()
selected_model = st.sidebar.selectbox(
    "Choose a model:",
    options=list(available_models.keys()),
    index=0  # Default to first model
)

st.sidebar.title("TFIDF Selection")
available_tfidf = load_tfidf_vectorizer()
selected_tfidf = st.sidebar.selectbox(
    "Choose a tfidf:",
    options=list(available_tfidf.keys()),
    index=0  # Default to first model
)

def load_model_components(model_name, selected_tfidf):
    st.subheader(model_name)
    model = available_models[model_name]
    vectorizer = joblib.load(f"{TFIDF_DIR}/{selected_tfidf}.pkl")
    label_mapping = joblib.load(f"{MODEL_DIR}/{model_name}.pkl")
    return model, vectorizer, label_mapping

# Preprocess the input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    return text

st.title("Text Classification Demo")
user_input = st.text_area("Enter text to classify:", "Type here")

# Add a button to trigger prediction
if st.button("Analyze"):
    try:
        model, vectorizer, labels = load_model_components(selected_model, selected_tfidf)
        # Preprocess input text (e.g., vectorize)
        cleaned_input = clean_text(user_input)
        # Transform using TF-IDF
        input_tfidf = vectorizer.transform([cleaned_input])
        
        # Predict
        prediction = model.predict(input_tfidf)
        prediction_proba = model.predict_proba(input_tfidf)
        
        # Map numerical class to label (3 classes)
        label_mapping = {
            0: 'Negative',
            1: 'Neutral',
            2: 'Positive'
        }

        metrics = {
            'sentiment': label_mapping[prediction[0]],
            'confidence': np.max(prediction_proba),
            'probabilities': {
                'negative': prediction_proba[0][0],
                'neutral': prediction_proba[0][1],
                'positive': prediction_proba[0][2]
            }
        }
        print(metrics)

        # Display result
        st.subheader("Prediction:")
        st.success(f"**{metrics['sentiment']}**")

        # Optional: Add confidence scores (if your model supports it)
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_tfidf)[0]
            st.write("Confidence Scores:")
            prob_df = pd.DataFrame({
                "Class": label_mapping.values(),
                "Probability": probabilities
            })
            st.bar_chart(prob_df.set_index("Class"))
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")