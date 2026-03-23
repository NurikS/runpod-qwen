import runpod
import logging
from pydantic import BaseModel, ValidationError
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

class TextRequest(BaseModel):
    text: str

def handler(event):
    try:
        request = TextRequest(**event["input"])
    except (ValidationError, KeyError) as e:
        logger.error(f"Invalid input: {e}")
        return {"error": f"Invalid input: {str(e)}"}

    try:
        messages = [{"role": "user", "content": request.text + " /no_think"}]
        result = pipe(messages)
        response = result[0]["generated_text"][-1]["content"]
        logger.info(f"Generated response for: {request.text[:50]}...")
        return {"response": response}
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {"error": f"Inference failed: {str(e)}"}

runpod.serverless.start({"handler": handler})
