import streamlit as st
from transformers import pipeline

# Load the translation pipeline (replace with your own model if needed)
translator = pipeline("translation_en_to_ur", model="Helsinki-NLP/opus-mt-en-ur")

# Title of the app
st.title("English to Roman Urdu Translator")

# Input text box
input_text = st.text_area("Enter English text:")

# Translate button
if st.button("Translate"):
    if input_text:
        # Translate input text to Roman Urdu
        translated_text = translator(input_text)[0]['translation_text']
        # Display the result
        st.success("Translated Text: ")
        st.write(translated_text)
    else:
        st.error("Please enter some text to translate.")

# Optional: Provide instructions or examples
st.sidebar.header("Instructions")
st.sidebar.text("1. Enter English text in the text area.\n"
                "2. Click 'Translate' to get the Roman Urdu translation.")
