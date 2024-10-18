import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer from Hugging Face
MODEL_NAME = "your-model-name"  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Streamlit app
st.title("English to Roman Urdu Translator")

# Input prompt from the user
input_text = st.text_area("Enter English text:")

if st.button("Translate"):
    # Encode input text
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate translation (in Roman Urdu)
    outputs = model.generate(inputs)
    
    # Decode and display the translated text
    roman_urdu_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.subheader("Roman Urdu Translation")
    st.write(roman_urdu_translation)
