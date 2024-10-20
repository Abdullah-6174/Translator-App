import streamlit as st
from transformers import pipeline
from googletrans import Translator

# Initialize the Hugging Face translator for English to Urdu
translator_model = pipeline("translation_en_to_ur", model="Helsinki-NLP/opus-mt-en-ur")

# Initialize the Google Translator for transliteration
transliterator = Translator()

# Streamlit app
def main():
    # Set title and description
    st.title("English to Roman Urdu Translator")
    st.write("Translate English text to Roman Urdu using translation and transliteration techniques.")

    # Input text from the user
    input_text = st.text_area("Enter English text:", "")

    # If the user inputs text, perform the translation and transliteration
    if st.button("Translate"):
        if input_text:
            # Perform English to Urdu translation
            urdu_translation = translator_model(input_text)[0]['translation_text']
            
            # Transliterate Urdu to Roman Urdu
            roman_urdu_translation = transliterator.translate(urdu_translation, src='ur', dest='ur', transliteration=True).text
            
            # Display the translations
            st.write("### Urdu Translation:")
            st.write(urdu_translation)
            st.write("### Roman Urdu Translation:")
            st.write(roman_urdu_translation)
        else:
            st.write("Please enter some text to translate.")

if __name__ == "__main__":
    main()
