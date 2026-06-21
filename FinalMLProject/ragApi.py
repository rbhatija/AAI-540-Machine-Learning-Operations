from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import RAGEngine
import time
import json
import os
from datetime import datetime

app = FastAPI()

engine = RAGEngine()

class Query(BaseModel):
    question: str

# ensure logs folder exists
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/rag_logs.jsonl"


def log_request(data):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")

@app.post("/ask")
def ask_api(query: Query):

    try:
        start = time.time()

        answer = engine.ask(query.question)
    
        latency = round(time.time() - start, 3)
    
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": query.question,
            "answer_preview": answer[:300],   # avoid huge logs
            "latency_sec": latency
        }
    
        log_request(log_entry)
    
        return {
            "answer": answer,
            "latency_sec": latency
        }

    except Exception as e:
        return {
            "error": str(e)
        }