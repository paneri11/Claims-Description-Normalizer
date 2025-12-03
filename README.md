Claims Description Normalizer (GenAI Project)
Convert messy insurance claim notes into clean, structured JSON using an offline LLM (GGUF + llama.cpp).

This project was built as part of a GenAI Engineer recruitment demo for ValueMomentum.
It demonstrates the ability to use NLP, entity extraction, schema design, prompt engineering, offline LLMs, and UI development to automate claim understanding.

🚀 Features
1. Offline LLM (Mistral 7B GGUF)

Runs with llama.cpp (no internet needed)

Supports air-gapped environments

Solves data-privacy concerns for insurers

2. Claim Normalization

Model extracts:

loss_type

severity (Low / Medium / High / Critical)

asset

estimated_loss

incident_date

location

confidence

explanation

Outputs clean JSON ready for downstream systems.

🖥️ Gradio Web Application

The app provides two main tabs:

🔍 Normalize Claim

Input messy claim text

View structured JSON

View readable summary

Use sample claims for quick demo

📊 History & Analytics

Every processed claim gets saved

View full history table

Bar charts:

Claims by severity

Claims by loss type

🏗️ Architecture
Raw Claim Text
      ↓
Prompt Engine (normalizer.py)
      ↓
Local GGUF LLM (llama.cpp via llama-cpp-python)
      ↓
JSON Schema Validator (Pydantic)
      ↓
UI + History Logging (Gradio + Pandas)

📦 Project Structure
claims-description-normalizer/
│
├── llm_engine.py          # Loads GGUF model
├── normalizer.py          # Prompt + JSON extraction
├── schema.py              # Pydantic model
├── ui_app.py              # Gradio UI (2 tabs)
│
├── models/                # GGUF model here
├── data/                  # History CSV auto-generated
├── requirements.txt
└── README.md

🧩 Setup & Run
1. Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

3. Download the GGUF model

Place it under models/

Recommended:

mistral-7b-instruct-v0.2.Q4_K_M.gguf

4. Run the app
python ui_app.py


Open:
http://127.0.0.1:7860

🧪 Model Evaluation (Mini Test)

A small evaluation script runs a batch of labelled claims and compares outputs.

(Section under development.)

💡 Future Enhancements

REST API (FastAPI)

Confidence scoring improvement

Fine-tuned insurance-specific LLM

Claim summarization + recommendations

Integration with claim management systems

👨‍💻 Author

Paneri Fulbandhe
B.Tech CSE – GenAI Engineer Aspirant
Project developed for ValueMomentum recruitment demo

📌 Status: READY FOR DEMO