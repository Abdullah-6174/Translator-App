# Install necessary libraries
pip install streamlit transformers

# Import necessary libraries
import streamlit as st
from transformers import pipeline

# Load the Hugging Face translation model
# For demonstration, I'm assuming you're using a model that translates English to Roman Urdu.
# Replace "translation_model_name" with the name of the actual model from Hugging Face.
@st.cache_resource
def load_model():
    model = pipeline("translation_en_to_ro_ur", model="translation_model_name")
    return model

# Streamlit app setup
st.title("English to Roman Urdu Translation")
st.write("Enter an English prompt below to get the translation in Roman Urdu:")

# Input box for user prompt
english_prompt = st.text_input("Enter your English text:")

# Load the model
translator = load_model()

# Perform translation when the user provides input
if english_prompt:
    translation = translator(english_prompt)[0]['translation_text']
    st.write(f"**Roman Urdu Translation:** {translation}")
