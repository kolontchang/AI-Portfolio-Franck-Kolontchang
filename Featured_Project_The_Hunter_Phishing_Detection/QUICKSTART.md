# The Hunter: Agentic Phishing Defense System

**ITAI 2376 - Final Project Delivery**  
**Team 1:** Chloe Tu, Mattew Choo, Franck Kolontchang

This repository contains the deployment components for "The Hunter," a fully-functional multi-agent phishing defense system featuring three LLM-driven CrewAI agents and an ensemble deep learning classifier.

## 📦 What's Included
* `hunter_gradio_app.py`: The live web dashboard built on Gradio.
* `working_the_hunter_notebook.ipynb`: The training architecture demonstrating the BiLSTM ensemble logic.
* `final_project_writeup.md`: Our team's architectural blueprint evaluation and results.

## 🚀 Quickstart Instructions (Local Machine)

The entire system is designed to boot with one command, provided you have configured the environment securely.

### 1. Install Dependencies
Open your terminal strictly inside this `Hunter Demo` folder and run:
`pip install -r requirements.txt`
*(Note: On Windows, utilize Command Prompt or PowerShell)*

### 2. Configure the `.env` Security Key
Because this application relies on live LLaMA 3.1 agents to orchestrate the phishing detection, it requires an API authentication key from Groq to function. **The agents will disable themselves and block the threat traces if this is missing.**

1. Open the `.env` file included in this directory in any text editor.
2. If you do not have a Groq account, visit [https://console.groq.com/](https://console.groq.com/) to quickly register and generate a free API key.
3. Paste the key precisely onto line 10 after the equals sign so it looks like this:
   `GROQ_API_KEY=gsk_yourKeyHere123abc`
4. **Save the file.**

### 3. Ensure Models Are Loaded
Ensure that a folder literally named `models` exists in this directory, and that it contains the pre-trained artifacts (`bilstm_model.keras`, etc.).
*(Note: If you downloaded the complete project package, these files are already in place!)*

### 4. Launch The System!
In your terminal, execute:
* **Mac/Linux:** `python3 hunter_gradio_app.py`
* **Windows:** `python hunter_gradio_app.py`

A local URL (e.g., `http://127.0.0.1:7860`) will be provided in the terminal output. Click the link to access the live dashboard!

---

## ☁️ Cloud Alternatives (If your computer cannot run it locally)

If your local computer experiences memory issues or you cannot install the libraries, you can run the Gradio interface smoothly in the cloud for free!

### Alternative A: Google Colab (Easiest)
1. Open a new Google Colab notebook.
2. Open the file sidebar and upload `hunter_gradio_app.py`, `requirements.txt`, and the entire `models/` folder.
3. Add your Groq API Key to Colab's built-in **Secrets** tab (Name it `GROQ_API_KEY`).
4. In a code cell, run `!pip install -r requirements.txt`.
5. In the next cell, run `!python hunter_gradio_app.py`.
Gradio will automatically generate a public `https://something.gradio.live` link directly in the cell output!

### Alternative B: Hugging Face Spaces (Most Professional)
1. Navigate to [HuggingFace Spaces](https://huggingface.co/spaces) and create a free Space (select Gradio as the Space SDK).
2. The platform will ask you to upload your files. Upload this entire repo exactly as it is (including the `models/` folder).
3. Go to your Space **Settings**, find **Variables and secrets**, and add a New Secret. Name it `GROQ_API_KEY` and paste your key.
4. Hugging Face will automatically detect `hunter_gradio_app.py`, install the requirements, and build a permanent URL for your dashboard!
