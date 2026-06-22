# Scripture RAG Assistant

## Project Overview

The Scripture RAG Assistant is a Retrieval-Augmented Generation (RAG) system designed to answer natural language questions based on two spiritual texts: the Bhagavad Gita and the Srimad Bhagavatam. The system retrieves relevant verses and passages from both scriptures and uses a language model to generate grounded, citation-backed responses.

The main objective of the system is to ensure that every generated answer is traceable to actual scripture passages rather than being generated from model memory alone. This improves factual grounding, interpretability, and trustworthiness.

The system is implemented as an end-to-end machine learning pipeline including data processing, embedding generation, vector search, LLM-based response generation, deployment on AWS SageMaker, monitoring, and CI/CD automation.

---

## System Architecture

The system follows a RAG architecture with two independent retrieval pipelines.

**Flow:**
- Raw scripture data stored in Amazon S3  
- Data preprocessing and cleaning (diacritics removal, chunking, verse segmentation)  
- Embedding generation using BAAI/bge-base-en-v1.5  
- Vector storage using FAISS indexes (separate indexes for BG and SB)  
- Query embedding and similarity search  
- Context injection into Qwen2.5-1.5B-Instruct LLM  
- Response generation with grounding constraints  
- FastAPI service exposed via SageMaker endpoint  

**Key Components:**
- Data Storage: Amazon S3  
- Feature Store Equivalent: FAISS indexes  
- Model Serving: AWS SageMaker endpoint + FastAPI  
- Embedding Model: BAAI/bge-base-en-v1.5  
- LLM: Qwen2.5-1.5B-Instruct  
- Search: FAISS IndexFlatIP (cosine similarity)  

---

## Project Structure

- `rag_engine.py` → Core RAG pipeline (retrieval + generation)
- `ragApi.py` → FastAPI service exposing `/ask` endpoint
- `sb_index.faiss / bg_index.faiss` → Vector indexes
- `sb_index_map.pkl / bg_index_map.pkl` → Metadata mappings
- `logs/` → Request and latency logs
- `.github/workflows/` → CI/CD pipeline

---

## How to Run the Project

### 1. Install dependencies
Make sure you have Python 3.10 installed, then run:

pip install -q torch torchvision torchaudio transformers accelerate sentence-transformers faiss-cpu tf-keras

### 2. Start the FastAPI server
uvicorn ragApi:app --host 0.0.0.0 --port 8000
The service will be available at:
http://127.0.0.1:8000

### 3. Test the API
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question":"What krishna instructs shall we ultimately surrender unto?"}'


---

## Features

- Dual knowledge base retrieval (Bhagavad Gita + Srimad Bhagavatam)
- Grounded answer generation using retrieved context only
- Independent FAISS indexes for each scripture
- Semantic search using dense embeddings
- End-to-end logging of latency and queries
- Deployed on AWS SageMaker real-time endpoint
- CI pipeline using GitHub Actions
- Basic monitoring using request logs

---

## Data Pipeline

### Data Sources
- Bhagavad Gita: 700 verses across 18 chapters  
- Srimad Bhagavatam: 24,329 text chunks across 12 cantos  
- Stored in Amazon S3  

### Preprocessing
- Unicode normalization (diacritics removal)  
- Removal of structural labels (PURPORT, Synonyms, etc.)  
- Verse-level segmentation for BG  
- Chunking (~800 characters) for SB  

### Embedding
- Model: BAAI/bge-base-en-v1.5  
- Output: 768-dimensional normalized embeddings  
- Stored in FAISS indexes using cosine similarity  

---

## Model Design

### Retrieval Model
- Algorithm: Dense vector similarity search  
- Library: FAISS (IndexFlatIP)  
- Strategy: Top-k retrieval per scripture  
- Expansion: Over-retrieval followed by ranking  

### Generation Model
- Model: Qwen2.5-1.5B-Instruct  
- Input: Question + retrieved context  
- Output: Grounded natural language response  
- Decoding: Greedy (deterministic output)  

---

## Deployment

- Platform: AWS SageMaker real-time endpoint  
- Instance Type: `ml.m3.xlarge` (CPU-based)  
- Serving Layer: FastAPI  
- Architecture: Synchronous request-response API  

Reason for design:
- Real-time question answering use case  
- Cost-optimized CPU deployment  
- Preloaded models for low-latency inference  

---

## Monitoring

Monitoring is implemented using log-based tracking.

**Tracked metrics:**
- Query timestamp
- User question
- Response preview
- Response latency

Stored in:
- `logs/rag_logs.jsonl`

**Current limitations:**
- No CloudWatch dashboard integration yet  
- No automated alerting system  
- No drift detection pipeline  

---

## CI/CD Pipeline

Implemented using GitHub Actions.

**Trigger:**
- Push to `main` branch  

**Steps:**
- Checkout repository  
- Setup Python environment  
- Install dependencies  
- Run import validation test  

**Validation test:**
- Ensures `rag_engine` imports successfully  

**Limitations:**
- No automated deployment to SageMaker  
- No unit tests or endpoint smoke tests  
- No rollback mechanism  

---

## Security and Risks

- No PHI or structured PII is collected  
- User queries may contain sensitive text stored in logs  
- Logs are stored locally on endpoint instance  
- Dataset reflects a single translation tradition (interpretation bias)  
- System may surface historical or cultural bias from source texts  

---

## Model Limitations

- CPU-based deployment leads to higher latency (~25–30 sec)  
- Context window limits long passages  
- Retrieval quality depends on embedding similarity only  
- No re-ranking layer implemented  

---

## Future Improvements

- Move to GPU-based SageMaker deployment for lower latency  
- Add re-ranking layer for improved retrieval quality  
- Add CloudWatch dashboards and alerting  
- Add PII redaction in logs  
- Replace FAISS with managed vector database  
- Build formal evaluation dataset (recall@k, MRR, faithfulness scoring)  
- Extend CI/CD to full deployment pipeline with Model Registry  

---

## Team Information

- Group: 2  
- Project: Scripture RAG Assistant  
- Authors: Rakesh Rajkumar Bhatija, Anusha Bandaru  
- Repository: https://github.com/rbhatija/AAI-540-Machine-Learning-Operations/tree/main/FinalMLProject  
