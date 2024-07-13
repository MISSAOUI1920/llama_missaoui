import streamlit as st
from peft import PeftModel, PeftConfig
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the model and tokenizer
new_model="llama-chat-guanaco"

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(new_model)
tokenizer = AutoTokenizer.from_pretrained(new_model)



# Streamlit app layout
st.title("Chatbot with LLaMA Model")
st.write("Enter your message below and get a response from the chatbot.")

user_input = st.text_input("You: ")

if st.button("Send"):
    if user_input:
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        
        with torch.no_grad():
            output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        st.write(f"Chatbot: {response}")
    else:
        st.write("Please enter a message.")
