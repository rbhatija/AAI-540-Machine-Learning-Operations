import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


class RAGEngine:

    def __init__(self):

        # ---------------- LOAD INDEXES ----------------
        self.sb_index = faiss.read_index("sb_index.faiss")
        self.bg_index = faiss.read_index("bg_index.faiss")

        self.sb_index_map = pickle.load(open("sb_index_map.pkl", "rb"))
        self.bg_index_map = pickle.load(open("bg_index_map.pkl", "rb"))

        # ---------------- DEVICE ----------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---------------- EMBEDDER ----------------
        self.embedder = SentenceTransformer(
            "BAAI/bge-base-en-v1.5",
            device=self.device
        )

        # ---------------- LLM ----------------
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    # ---------------- CLEAN ----------------
    def clean(self, text, max_chars=900):
        return text.replace("\n", " ")[:max_chars]

    # ---------------- BG SEARCH ----------------
    def search_bg(self, question, k=2):

        q_emb = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores, ids = self.bg_index.search(q_emb, k * 10)

        results = []

        for score, idx in zip(scores[0], ids[0]):

            doc = self.bg_index_map[idx]

            results.append({
                "score": float(score),
                "chapter": doc.get("chapter"),
                "verse_id": doc.get("verse_id"),
                "text": doc["text"]
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    # ---------------- SB SEARCH ----------------
    def search_sb(self, question, k=2):

        q_emb = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores, ids = self.sb_index.search(q_emb, k * 10)

        results = []

        for rank, (idx, score) in enumerate(zip(ids[0], scores[0])):

            doc = self.sb_index_map[idx]

            results.append({
                "rank": rank + 1,
                "score": float(score),
                "chapter": doc.get("chapter"),
                "canto": doc.get("canto"),
                "text": doc["text"]
            })

        # IMPORTANT: keep best scoring first
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:k]

    # ---------------- ASK ----------------
    def ask(self, question):

        bg_results = self.search_bg(question)
        sb_results = self.search_sb(question)

        bg_context = "\n\n".join(self.clean(r["text"]) for r in bg_results)
        sb_context = "\n\n".join(self.clean(r["text"]) for r in sb_results)

        # IMPORTANT FIX: stronger grounding instruction (WITHOUT changing your style)
        prompt = f"""
You are a spiritual knowledge assistant.

Use ONLY the provided context.
You MUST include BOTH Bhagavad Gita and Srimad Bhagavatam insights.

Do not ignore SB context.

QUESTION:
{question}

BG CONTEXT:
{bg_context}

SB CONTEXT:
{sb_context}

ANSWER:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            out = self.llm.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False
            )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # FIX: prevent prompt leakage safely
        answer = decoded.split("ANSWER:")[-1].strip()

        # ---------------- KEEP YOUR FORMAT STYLE ----------------
        bg_block = ""
        if bg_results:
            bg = bg_results[0]
            bg_block = f"""
📜 BHAGAVAD GITA

Chapter: {bg.get('chapter', '?')}
Verse: {bg.get('verse_id', '?')}

Text:
{self.clean(bg['text'])[:400]}...
"""

        sb_block = "\n\n".join([
            f"""
📖 SRIMAD BHAGAVATAM

Canto: {s.get('canto', '?')}
Chapter: {s.get('chapter', '?')}

Text:
{self.clean(s['text'])[:250]}...
"""
            for s in sb_results
        ])

        return f"""
══════════════════════════════════════
QUESTION:
{question}

{bg_block}

{sb_block}

🎯 FINAL ANSWER:
{answer}

══════════════════════════════════════
"""
