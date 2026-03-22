FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN pip install runpod transformers

COPY handler.py /handler.py

CMD ["python", "/handler.py"]
