import streamlit as st
from transformers import pipeline

# Load the Hugging Face translation pipeline for English to Urdu
translator_model = pipeline("translation_en_to_ur", model="Helsinki-NLP/opus-mt-en-ur")

# Custom function for Urdu to Roman Urdu transliteration
def urdu_to_roman_urdu(urdu_text):
    # Basic transliteration mapping (extend this for better accuracy)
    mapping = {
        'ا': 'a', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 'ṭ', 'ث': 'th',
        'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'ḍ',
        'ذ': 'dh', 'ر': 'r', 'ڑ': 'ṛ', 'ز': 'z', 'ژ': 'zh', 'س': 's',
        'ش': 'sh', 'ص': 'ṣ', 'ض': 'z̤', 'ط': 'ṭ', 'ظ': 'ẓ', 'ع': '‘',
        'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g', 'ل': 'l',
        'م': 'm', 'ن': 'n', 'ں': 'ñ', 'و': 'w', 'ہ': 'h', 'ء': 'ʾ',
        'ی': 'y', 'ے': 'e', '؟': '?', '،': ',', '۔': '.'
    }
    # Transliterate Urdu text to Roman Urdu using the mapping
    roman_urdu_text = ''.join([mapping.get(char, char) for char in urdu_text])
    return roman_urdu_text

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
            
            # Perform Urdu to Roman Urdu transliteration
            roman_urdu_translation = urdu_to_roman_urdu(urdu_translation)
            
            # Display the translations
            st.write("### Urdu Translation:")
            st.write(urdu_translation)
            st.write("### Roman Urdu Translation:")
            st.write(roman_urdu_translation)
        else:
            st.write("Please enter some text to translate.")

if __name__ == "__main__":
    main()
