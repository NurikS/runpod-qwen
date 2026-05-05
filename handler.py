import runpod
import logging
import os
import io
import base64
import numpy as np
import faiss
from pydantic import BaseModel, ValidationError
from transformers import pipeline, AutoTokenizer, AutoModel, BlipProcessor, BlipForConditionalGeneration
import torch
from typing import List, Tuple, Optional, Dict
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== RAG Configuration =====
DOCS_DIR = "/docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension
MIN_IMAGE_SIZE = 100  # Minimum image dimension to process (skip tiny icons)

# ===== Load Models =====
logger.info("Loading LLM...")
pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

logger.info("Loading embedding model...")
embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

logger.info("Loading BLIP image captioning model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ===== Document Store =====
doc_chunks: List[str] = []
chunk_images: Dict[int, str] = {}  # Maps chunk index to base64 image data
faiss_index: Optional[faiss.IndexFlatIP] = None  # Inner product (cosine sim for normalized vectors)


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


def caption_image(image: Image.Image) -> str:
    """Generate a caption for an image using BLIP."""
    try:
        inputs = blip_processor(image, return_tensors="pt")
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_length=50)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logger.error(f"Failed to caption image: {e}")
        return ""


def extract_images_from_pdf(filepath: str) -> List[Tuple[Image.Image, int]]:
    """Extract images from PDF, returns list of (image, page_number)."""
    import fitz
    images = []
    
    try:
        doc = fitz.open(filepath)
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    
                    # Skip tiny images (likely icons or decorations)
                    if image.width >= MIN_IMAGE_SIZE and image.height >= MIN_IMAGE_SIZE:
                        images.append((image, page_num + 1))
                        
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                    
        doc.close()
    except Exception as e:
        logger.error(f"Failed to extract images from {filepath}: {e}")
    
    return images


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def load_documents():
    """Load and index all documents from DOCS_DIR."""
    global doc_chunks, chunk_images, faiss_index
    
    if not os.path.exists(DOCS_DIR):
        logger.warning(f"Docs directory {DOCS_DIR} does not exist")
        return
    
    all_chunks = []
    image_data = {}  # Temporary storage for image base64
    
    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        
        try:
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                chunks = chunk_text(text)
                logger.info(f"Loaded {filename}: {len(chunks)} text chunks")
                all_chunks.extend(chunks)
                
            elif filename.endswith('.pdf'):
                import fitz  # PyMuPDF
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                
                # Process text chunks
                chunks = chunk_text(text)
                logger.info(f"Loaded {filename}: {len(chunks)} text chunks")
                all_chunks.extend(chunks)
                
                # Extract and caption images
                images = extract_images_from_pdf(filepath)
                logger.info(f"Found {len(images)} images in {filename}")
                
                for img, page_num in images:
                    caption = caption_image(img)
                    if caption:
                        # Store the chunk index before adding
                        chunk_idx = len(all_chunks)
                        # Create a chunk from the image caption with metadata
                        image_chunk = f"[Image on page {page_num}]: {caption}"
                        all_chunks.append(image_chunk)
                        # Store the base64 image data
                        image_data[chunk_idx] = image_to_base64(img)
                        logger.info(f"  Page {page_num}: {caption[:50]}...")
                
            else:
                continue
            
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
    
    if all_chunks:
        doc_chunks = all_chunks
        chunk_images = image_data  # Store image data globally
        logger.info(f"Generating embeddings for {len(doc_chunks)} chunks...")
        embeddings = [get_embedding(chunk) for chunk in doc_chunks]
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings_array)
        
        # Create FAISS index
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        faiss_index.add(embeddings_array)
        
        logger.info(f"FAISS index ready: {faiss_index.ntotal} vectors, {len(chunk_images)} images")
    else:
        logger.warning("No documents found to index")


def search_docs(query: str, top_k: int = TOP_K) -> List[Tuple[int, str, float]]:
    """Search documents and return top-k relevant chunks using FAISS.
    Returns list of (chunk_index, chunk_text, score)."""
    if faiss_index is None or len(doc_chunks) == 0:
        return []
    
    query_embedding = get_embedding(query).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    # FAISS search
    scores, indices = faiss_index.search(query_embedding, min(top_k, len(doc_chunks)))
    
    results = [(int(idx), doc_chunks[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
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
        
        # Collect images from matched chunks
        images = []
        for idx, doc, score in relevant_docs:
            if idx in chunk_images:
                # Extract page number from chunk text
                page_match = doc.split("]")[0].replace("[Image on page ", "")
                try:
                    page_num = int(page_match)
                except:
                    page_num = 0
                images.append({
                    "data": chunk_images[idx],
                    "page": page_num,
                    "caption": doc.split("]: ", 1)[-1] if "]: " in doc else doc
                })
        
        if relevant_docs:
            context = "\n\n".join([f"[Relevant excerpt {i+1}]:\n{doc}" for i, (idx, doc, score) in enumerate(relevant_docs)])
            
            # If there are images, instruct LLM to explain them
            if images:
                prompt = f"""Based on the following documentation excerpts (including image descriptions), answer the user's question.
The user will see the actual images displayed above your response, so explain what the diagrams show and how they relate to the question.

Documentation:
{context}

Question: {query}

First explain what the relevant diagram(s) show, then answer the question based on the documentation."""
            else:
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
        
        logger.info(f"Generated response for: {query[:50]}... (found {len(relevant_docs)} chunks, {len(images)} images)")
        return {"response": response, "images": images}
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {"error": f"Inference failed: {str(e)}"}


runpod.serverless.start({"handler": handler})
