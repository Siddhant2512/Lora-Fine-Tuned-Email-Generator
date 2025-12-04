"""FastAPI backend for email generation."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model_loader import EmailModel
from src.config import load_config

app = FastAPI(title="LoRA-Mail Assistant API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
config = load_config()

# Initialize model (lazy loading)
_model: Optional[EmailModel] = None


def get_model() -> EmailModel:
    """Lazy load model on first request."""
    global _model
    if _model is None:
        model_path = config['training']['output_dir']
        _model = EmailModel(
            model_path=model_path if Path(model_path).exists() else None,
            use_quantization=config['inference']['use_quantization']
        )
    return _model


class EmailRequest(BaseModel):
    recipient: str
    purpose: str
    key_points: str
    tone: str = "professional"


class EmailResponse(BaseModel):
    email: str
    model_used: str


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "LoRA-Mail Assistant API",
        "model": config['model']['base_model']
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/generate", response_model=EmailResponse)
async def generate_email(request: EmailRequest):
    """Generate an email based on the provided context."""
    try:
        model = get_model()
        email = model.generate_email(
            recipient=request.recipient,
            purpose=request.purpose,
            key_points=request.key_points,
            tone=request.tone
        )
        
        return EmailResponse(
            email=email,
            model_used=config['model']['base_model']
        )
    except RuntimeError as e:
        # Handle authentication errors
        if "authentication" in str(e).lower():
            raise HTTPException(
                status_code=401,
                detail=f"Authentication required: {str(e)}. Run: huggingface-cli login"
            )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        error_msg = str(e)
        # Provide helpful error messages
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
            error_msg += ". Try reducing batch size or using a smaller model."
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    import uvicorn
    api_config = config['api']
    uvicorn.run(app, host=api_config['host'], port=api_config['port'])

