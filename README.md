# VorenLite AI Assistant

> A lightweight, conversational AI assistant powered by an efficient 360M parameter language model.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41.0-FF4B4B?logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-4.49.0-FFD21E)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)

## Overview

**VorenLite AI Assistant** is a streamlined, resource-efficient conversational AI web application built to demonstrate practical Large Language Model (LLM) implementation without relying on external paid APIs. 

Powered by Hugging Face's **`SmolLM2-360M-Instruct`**, this project showcases how compact, instruction-tuned causal language models can run smoothly on standard CPU hardware while maintaining coherent, multi-turn dialogue through structured chat templates and session state management.

---

## Key Features

- **Lightweight Local Inference:** Runs entirely on CPU using `SmolLM2-360M-Instruct` (`float32`), eliminating API latency and third-party costs.
- **Multi-Turn Context Awareness:** Utilizes Streamlit's `st.session_state` alongside a sliding context window (system prompt + recent conversation turns) to maintain coherent conversations without exceeding token limits.
- **Modern Chat Templating:** Implements Hugging Face's `tokenizer.apply_chat_template()` to properly format `system`, `user`, and `assistant` roles.
- **Optimized Resource Caching:** Uses `@st.cache_resource` to load model weights into memory once per session, preventing redundant reloads across user interactions.
- **Tuned Text Generation:** Configured with balanced sampling parameters (`temperature`, `top_p`, `repetition_penalty`, and `no_repeat_ngram_size`) for natural, non-repetitive responses.

---

## Tech Stack

- **Language:** Python 3.12
- **Frontend & Web Framework:** Streamlit (`1.41.0`)
- **Deep Learning & Inference:** PyTorch (`2.5.1`), Accelerate (`1.2.1`)
- **NLP & Model Pipeline:** Hugging Face Transformers (`4.49.0`)
- **Model:** [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)

---

## Getting Started

### Prerequisites

- Python **3.12** installed on your system
- `pip` package manager

### Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GITvoren/chatbot-lite.git
   cd chatbot-lite
   ```

2. **Create and activate a virtual environment:**
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Or install directly via pip)*:
   ```bash
   pip install transformers==4.49.0 torch==2.5.1 accelerate==1.2.1 streamlit==1.41.0 --prefer-binary
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```
   The app will automatically open in your default browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
chatbot-lite/
├── app.py               # Main Streamlit application & LLM inference pipeline
├── requirements.txt     # Pinned Python dependencies for reproducible deployment
└── README.md            # Project documentation
```

---

### Author & Course Context
Built by **Voren** ([GitHub](https://github.com/GITvoren) • [Portfolio](https://vorentejuco.pages.dev))

> **Part of the IBM AI Developer Professional Certificate**  
> Developed as a practical, hands-on capstone/lab project applying modern AI engineering, LLM integration, and full-stack deployment principles.