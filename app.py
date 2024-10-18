import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from transformers import Trainer, TrainingArguments


# Load the dataset
dataset = load_dataset("Alisaeed001/EnglishToRomanUrdu")

# Load the tokenizer and model
model_name = "t5-small"  # You can choose a larger model if needed
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocess the dataset
def preprocess_function(examples):
    inputs = []
    targets = []

    for text in examples['text']:  # Assuming the loaded dataset has a column 'text'
        # Split the text using the appropriate markers
        if "[INST]" in text and "[/INST]" in text:
            english_part = text.split("[INST]")[1].split("[/INST]")[0].strip()
            roman_urdu_part = text.split("[/INST]")[1].split("</s>")[0].strip()  # Remove the </s> tag
            inputs.append(english_part)
            targets.append(roman_urdu_part)

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("roman_urdu_model")
tokenizer.save_pretrained("roman_urdu_model")

# Streamlit app remains the same
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
model_name = "roman_urdu_model"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define a function to convert English to Roman Urdu
def translate_to_roman_urdu(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(input_ids)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Streamlit app
st.title("English to Roman Urdu Translator")
st.write("Enter English text to convert it to Roman Urdu:")

# Input text box
user_input = st.text_area("Input English text")

# Translate button
if st.button("Translate"):
    if user_input:
        translation = translate_to_roman_urdu(user_input)
        st.success(f"Roman Urdu: {translation}")
    else:
        st.error("Please enter some text to translate.")
