FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN pip install runpod transformers sentence-transformers PyMuPDF numpy pydantic faiss-cpu Pillow

# Create docs directory
RUN mkdir -p /docs

# Copy handler and documentation
COPY handler.py /handler.py
COPY docs/ /docs/

CMD ["python", "/handler.py"]
