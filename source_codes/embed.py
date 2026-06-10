import json
import numpy as np
from sentence_transformers import SentenceTransformer

AIMS_AND_SCOPE = """
Natural Language Engineering is an open access journal which meets the needs 
of professionals and researchers working in all areas of natural language 
processing (NLP). Its aim is to bridge the gap between traditional 
computational linguistics research and the implementation of practical 
applications with potential real-world use. The journal publishes original 
research articles on a broad range of methods and resources applied in NLP, 
language processing tasks and NLP applications, including machine translation, 
translation technology, sentiment analysis, information retrieval, question 
answering, text summarisation, text simplification, and speech processing.
"""

MODEL_NAME = "all-MiniLM-L6-v2"
PAPERS_FILE = "papers.json"

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Encoding Aims & Scope...")
aims_embedding = model.encode(AIMS_AND_SCOPE.strip(), normalize_embeddings=True)
np.save("aims_embedding.npy", aims_embedding)
print("Aims & Scope embedding saved.")

print("Loading papers...")
with open(PAPERS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

papers = data["papers"]
abstracts = [p["abstract"] for p in papers]
print(f"Encoding {len(abstracts)} abstracts...")

embeddings = model.encode(abstracts, normalize_embeddings=True, show_progress_bar=True)
np.save("paper_embeddings.npy", embeddings)
print(f"Done. Embeddings saved: {embeddings.shape}")