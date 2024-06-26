import transformers
import streamlit as st
import re
from transformers import LlamaTokenizer, LlamaForCausalLM

# Cache the model and tokenizer loading to ensure they are only loaded once
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained('klyang/MentaLLaMA-chat-7B')
    model = LlamaForCausalLM.from_pretrained('klyang/MentaLLaMA-chat-7B', device_map='auto')
    return tokenizer, model
tokenizer, model = load_model_and_tokenizer()

# Define the prediction function
def predict(input_text):
    input_data = [f"Consider this post: {input_text}. Question: Does the poster suffer from depression?"]
    inputs = tokenizer(input_data, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to('cuda')
    generate_ids = model.generate(input_ids=input_ids, max_length=512)
    trunc_ids = generate_ids[0][len(input_ids[0]):]
    response = tokenizer.decode(trunc_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
    return response

# Streamlit app
st.title("Depression Detection App")
st.write("Provide a post or message to determine if it suggests the user may be suffering from depression.")

input_text = st.text_area("Enter a post/input text")

if st.button("Predict"):
    with st.spinner("Analyzing..."):
        output_text = predict(input_text)
        # Splitting the text into sentences
        sentences = output_text.strip().split('.')

        # Extracting the last sentence for the verdict
        verdict = sentences[-2].strip()

        # Extracting the last 3rd to last 2nd sentences for the reason
        reason = sentences[-4:-2]

        # Remove "Reasoning:" if it appears in the last two lines
        reason = [re.sub(r'reasoning:\s*', '', line, flags=re.IGNORECASE).strip() for line in reason]

        # Displaying the verdict and reason
        st.write("Verdict:")
        st.write(verdict)
        st.write("Reason:")
        st.write(". ".join(reason))
