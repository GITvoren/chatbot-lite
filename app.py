import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="VorenLite AI Assistant - SmolLM2-360M model", page_icon="🤖", layout="centered"
)

st.title("🤖 VorenLite AI Assistant")
st.caption(
    "A lightweight, conversational AI assistant powered by an efficient 360M parameter language model"
)

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"


# Load model and tokenizer with caching so it only loads once in Streamlit
@st.cache_resource
def load_model():
  print("Loading model...")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.pad_token = tokenizer.unk_token

  model = AutoModelForCausalLM.from_pretrained(
      model_name, device_map="cpu", torch_dtype=torch.float32
  )
  return tokenizer, model


with st.spinner("Loading SmolLM2 model weights... Please wait."):
  tokenizer, model = load_model()

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
  st.session_state.messages = [{
      "role": "system",
      "content": (
          "You are a helpful AI assistant. Give short and concise answers in"
          " 2-3 lines."
      ),
  }]

# Display prior chat messages (skipping the system prompt)
for message in st.session_state.messages:
  if message["role"] != "system":
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

# Accept user input via Streamlit's chat input bar (replaces input("> "))
if user_input := st.chat_input("Ask me anything..."):
  # Append user message to history and display it
  st.session_state.messages.append({"role": "user", "content": user_input})
  with st.chat_message("user"):
    st.markdown(user_input)

  # Keep system prompt + last 10 messages
  messages_to_process = [st.session_state.messages[0]] + st.session_state.messages[
      -10:
  ]

  # Apply chat template
  tokenized = tokenizer.apply_chat_template(
      messages_to_process,
      tokenize=True,
      add_generation_prompt=True,
      return_tensors="pt",
      return_dict=True,
      max_length=512,
  )

  # Generate bot response
  with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
      with torch.inference_mode():
        outputs = model.generate(
            tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=100,
            temperature=0.5,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )

      response = tokenizer.decode(
          outputs[0][tokenized["input_ids"].shape[-1] :],
          skip_special_tokens=True,
      )

      st.markdown(response)

  # Save assistant response to chat history
  st.session_state.messages.append({"role": "assistant", "content": response})