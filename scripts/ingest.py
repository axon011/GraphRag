# Simple ingestion: reads text/markdown files, splits, embeds, builds FAISS index.

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import faiss
import pickle

DATA_DIR = Path(r"c:\Users\Aravind\Desktop\workspace\GIT\graph\data")
INDEX_PATH = Path(r"c:\Users\Aravind\Desktop\workspace\GIT\graph\index.faiss")
META_PATH = Path(r"c:\Users\Aravind\Desktop\workspace\GIT\graph\meta.pkl")

def load_text_files(d):
    docs = []
    for p in d.rglob("*.md"):
        txt = p.read_text(encoding="utf-8")
        docs.append((str(p), txt))
    for p in d.rglob("*.txt"):
        docs.append((str(p), p.read_text(encoding="utf-8")))
    return docs

def clean_text(s):
    # minimal cleaning; add more rules as needed
    return " ".join(tok for tok in s.split() if tok not in ENGLISH_STOP_WORDS)

def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_index(docs):
    model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight
    embeddings = []
    metadatas = []
    for path, text in docs:
        text = clean_text(text)
        for chunk in chunk_text(text):
            emb = model.encode(chunk)
            embeddings.append(emb)
            metadatas.append({"source": path, "text": chunk})
    import numpy as np
    X = np.vstack(embeddings).astype("float32")
    dim = X.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metadatas, f)
    print(f"Saved index ({X.shape[0]} vectors) -> {INDEX_PATH}")

if __name__ == "__main__":
    docs = load_text_files(DATA_DIR)
    build_index(docs)