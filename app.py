import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import evaluate  # Import evaluate instead of load_metric
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Load the dataset
dataset = load_dataset("Alisaeed001/EnglishToRomanUrdu")

# Load the tokenizer and model
model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Preprocessing function for tokenizing
def preprocess_function(examples):
    inputs = examples["English"]
    targets = examples["Roman Urdu"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # Tokenize the target with the `text_target` keyword argument
    labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set training arguments
batch_size = 8
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
)

# Define metric for evaluation
metric = evaluate.load("sacrebleu")  # Use evaluate.load to load the metric

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = [[(label if label != -100 else tokenizer.pad_token_id) for label in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU score
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("fine-tuned-english-to-roman-urdu")
tokenizer.save_pretrained("fine-tuned-english-to-roman-urdu")

# Load the fine-tuned model and tokenizer
model_name = "fine-tuned-english-to-roman-urdu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Streamlit app
st.title("English to Roman Urdu Translator")

# Input prompt from the user
input_text = st.text_area("Enter English text:")

if st.button("Translate"):
    # Tokenize input text
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # Generate translation
    with torch.no_grad():
        outputs = model.generate(inputs)

    # Decode and display the translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.subheader("Roman Urdu Translation")
    st.write(translation)
