RAG-Based AI System (FastAPI + FAISS + Transformers)

This project implements a Retrieval-Augmented Generation (RAG) system using:

FAISS vector search
Sentence Transformers embeddings
HuggingFace LLM (Qwen)
FastAPI backend
Simple logging + monitoring
CI/CD via GitHub Actions
Project Structure
AAI-540-Machine-Learning-Operations/
│
├── FinalMLProject/
│   ├── ragApi.py
│   ├── rag_engine.py
│   ├── sb_index.faiss
│   ├── bg_index.faiss
│   ├── *.pkl files
│
├── .github/workflows/
│   └── deploy.yml
│
├── requirements.txt
└── README.md

Step 1: Install Dependencies

Open Jupyter Lab → Terminal and run:

pip install -q torch torchvision torchaudio transformers accelerate sentence-transformers faiss-cpu tf-keras
Step 2: Clone Repository
git clone https://github.com/rbhatija/AAI-540-Machine-Learning-Operations.git
cd AAI-540-Machine-Learning-Operations
Step 3: Run Deployment (FastAPI Server)

Open Terminal 1 and start the API:

uvicorn FinalMLProject.ragApi:app --host 0.0.0.0 --port 8000
Test API (Terminal 2)

Run curl command:

curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question":"What is karma yoga?"}'

You will receive a context-aware answer generated from Bhagavad Gita + Srimad Bhagavatam.

Step 4: Monitoring

Once the API starts receiving requests:

A logs/ folder is automatically created
Each request is logged with:
timestamp
question
response preview
latency

You can monitor system performance directly from these logs.

Step 5: CI/CD Pipeline

This project includes GitHub Actions CI/CD pipeline.

Trigger:

Every push to main branch triggers:

Code checkout
Dependency installation
Import validation test
Pipeline Name:
RAG CI/CD Pipeline
Where to check:

GitHub Repository → Actions tab

Features
Semantic retrieval using FAISS
Context-aware LLM responses
Dual knowledge base (BG + SB texts)
Structured API using FastAPI
Lightweight monitoring via logs
Automated CI pipeline via GitHub Actions
Notes
Ensure .github/workflows/deploy.yml is in repository root
Ensure requirements.txt is in root for CI to work
Run API from project root for correct file paths

Future Improvements
Docker containerization
Cloud deployment (EC2 / SageMaker endpoint)
Advanced monitoring (Prometheus + Grafana)
Feedback loop for RAG improvement
