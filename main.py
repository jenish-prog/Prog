from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_API_KEY")
if not HF_TOKEN:
    print("WARNING: HF_API_KEY not found in .env")

hf_client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None

app = FastAPI(title="Voice Assistant Cloud Backend")

MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-72B-Instruct",
]


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    if not hf_client:
        return {"error": "HF_API_KEY not configured"}

    messages = [
        {"role": "system", "content": "You are a helpful voice assistant. Keep your response concise, natural, and conversational. Do not use markdown."},
        {"role": "user", "content": req.message},
    ]

    for model in MODELS:
        try:
            response = hf_client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=256,
            )
            reply = response.choices[0].message.content
            return {"response": reply, "model": model}
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                continue
            elif "503" in error_str or "loading" in error_str.lower():
                continue
            else:
                return {"error": str(e)}

    return {"error": "All models are currently unavailable. Try again later."}


@app.get("/health")
def health():
    return {"status": "ok"}
