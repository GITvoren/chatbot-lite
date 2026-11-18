# py -3.12 -m venv venv
# python -m pip install --upgrade pip
# pip install transformers==4.41.2 torch==2.2.2 accelerate==0.30.1 numpy==1.26.4


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Choose the model optimized for dialogue
model_name = "facebook/blenderbot-400M-distill"

print("Loading model and tokenizer...")
# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Chatbot ready! (type 'exit' to quit)\n")

# Initialize conversation history list
conversation_history = []

# Main conversation loop
while True:
    # Keep only the last few exchanges to prevent confusion and stay within token limits
    conversation_history = conversation_history[-6:]
    history_string = "\n".join(conversation_history)

    # Fetch prompt from user
    input_text = input("> ")

    # Check if user wants to exit
    if input_text.lower() == "exit":
        print("Goodbye!")
        break

    # Combine history and new input into a prompt and tokenize it
    prompt = history_string + f"\nUser: {input_text}\nBot:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Generate output from the model
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        do_sample=True,
        temperature=0.6,
        top_p=0.85
    )
    
    # print(outputs)

    # Decode output into readable text

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("Bot:", response)

    # Update conversation history
    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"Bot: {response}")

 