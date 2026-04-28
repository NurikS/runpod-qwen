import runpod
import logging
import os
import numpy as np
from pydantic import BaseModel, ValidationError
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== RAG Configuration =====
DOCS_DIR = "/docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# ===== Load Models =====
logger.info("Loading LLM...")
pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

logger.info("Loading embedding model...")
embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# ===== Document Store =====
doc_chunks: List[str] = []
doc_embeddings: np.ndarray = None


def mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embedding(text: str) -> np.ndarray:
    """Get embedding for a single text."""
    encoded = embed_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        output = embed_model(**encoded)
    embedding = mean_pooling(output, encoded['attention_mask'])
    return embedding.numpy().flatten()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def load_documents():
    """Load and index all documents from DOCS_DIR."""
    global doc_chunks, doc_embeddings
    
    if not os.path.exists(DOCS_DIR):
        logger.warning(f"Docs directory {DOCS_DIR} does not exist")
        return
    
    all_chunks = []
    
    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        
        try:
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif filename.endswith('.pdf'):
                import fitz  # PyMuPDF
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
            else:
                continue
            
            chunks = chunk_text(text)
            logger.info(f"Loaded {filename}: {len(chunks)} chunks")
            all_chunks.extend(chunks)
            
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
    
    if all_chunks:
        doc_chunks = all_chunks
        logger.info(f"Generating embeddings for {len(doc_chunks)} chunks...")
        embeddings = [get_embedding(chunk) for chunk in doc_chunks]
        doc_embeddings = np.array(embeddings)
        logger.info(f"Document index ready: {len(doc_chunks)} chunks")
    else:
        logger.warning("No documents found to index")


def search_docs(query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
    """Search documents and return top-k relevant chunks."""
    if doc_embeddings is None or len(doc_chunks) == 0:
        return []
    
    query_embedding = get_embedding(query)
    
    # Cosine similarity
    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-9
    )
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = [(doc_chunks[i], float(similarities[i])) for i in top_indices]
    return results


# ===== Load documents on startup =====
logger.info("Loading documents...")
load_documents()


class TextRequest(BaseModel):
    text: str


def handler(event):
    try:
        request = TextRequest(**event["input"])
    except (ValidationError, KeyError) as e:
        logger.error(f"Invalid input: {e}")
        return {"error": f"Invalid input: {str(e)}"}

    try:
        query = request.text
        
        # RAG: Search relevant documents
        relevant_docs = search_docs(query)
        
        if relevant_docs:
            context = "\n\n".join([f"[Relevant excerpt {i+1}]:\n{doc}" for i, (doc, score) in enumerate(relevant_docs)])
            prompt = f"""Based on the following documentation excerpts, answer the user's question.

Documentation:
{context}

Question: {query}

Answer based on the documentation above. If the documentation doesn't contain relevant information, say so."""
        else:
            prompt = query
        
        messages = [{"role": "user", "content": prompt + " /no_think"}]
        result = pipe(messages)
        response = result[0]["generated_text"][-1]["content"]
        
        logger.info(f"Generated response for: {query[:50]}... (found {len(relevant_docs)} relevant chunks)")
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {"error": f"Inference failed: {str(e)}"}


runpod.serverless.start({"handler": handler})
