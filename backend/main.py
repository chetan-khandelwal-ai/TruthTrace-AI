
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai

app = FastAPI(title="TruthTrace AI API")

# Website permissions
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chetan-khandelwal-ai.github.io"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Secure API key from environment
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY) if API_KEY else None

MODEL_FALLBACKS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite"
]

class AnalysisRequest(BaseModel):
    text: str = Field(min_length=10, max_length=10000)


@app.get("/")
def home():
    return {
        "project": "TruthTrace AI",
        "status": "Backend is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/analyze")
def analyze(request: AnalysisRequest):

    if not client:
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured"
        )

    prompt = f"""
You are TruthTrace AI, an information
credibility analysis assistant.

Analyze the following content carefully.

CONTENT:
{request.text}

Provide:
1. Main claim or message
2. Potential misinformation indicators
3. Supporting evidence required
4. Missing context
5. A cautious credibility assessment

Do not invent facts, sources or evidence.
Clearly distinguish verified facts from uncertainty.
Do not claim that something is fake without evidence.
"""

    
models_to_try = [
    MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash"
]

last_error = None

for model_name in models_to_try:
    try:
        print(f"Trying AI model: {model_name}", flush=True)

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        if response.text:
            return {
                "success": True,
                "analysis": response.text,
                "model_used": model_name
            }

        last_error = "Empty AI response"

    except Exception as e:
        last_error = e

        print(
            f"MODEL FAILED: {model_name}: {type(e).__name__}: {e}",
            flush=True
        )

raise HTTPException(
    status_code=503,
    detail="All available AI models are temporarily unavailable. Please retry."
)

