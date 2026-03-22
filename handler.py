import runpod
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

def handler(event):
    text = event["input"]["text"]
    messages = [{"role": "user", "content": text}]
    result = pipe(messages)
    return {"response": result[0]["generated_text"][-1]["content"]}

runpod.serverless.start({"handler": handler})
