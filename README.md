# LoRA-Mail Assistant

A web-deployable AI assistant that fine-tunes a small open-source LLM using LoRA to write emails in your specific personal/professional style.

## Features

- **LoRA Fine-tuning**: Efficiently fine-tune a small LLM on your email writing style
- **M3 Mac Optimized**: Uses MPS acceleration and quantization for fast local inference
- **Web Interface**: Streamlit frontend for easy interaction
- **API Backend**: FastAPI backend for model inference
- **One-Task Mastery**: Specialized for email drafting

## Project Structure

```
LoRA-Mail Assistant/
├── api/
│   └── main.py              # FastAPI backend
├── ui/
│   └── streamlit_app.py     # Streamlit frontend
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── model_loader.py      # Model loading and inference
│   └── train.py             # LoRA fine-tuning script
├── data/
│   ├── examples/            # Example emails (JSONL format)
│   └── train_dataset.jsonl  # Training dataset
├── models/                  # Saved models (created after training)
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Hugging Face Authentication (Required for Llama 3.2 3B)

```bash
# Option 1: Use the helper script
python scripts/setup_hf_auth.py

# Option 2: Manual login
huggingface-cli login
```

Get your token from: https://huggingface.co/settings/tokens

### 4. Verify Setup

```bash
python scripts/check_setup.py
```

This will check:
- ✅ All dependencies installed
- ✅ Configuration valid
- ✅ Training data present
- ✅ Hugging Face authentication (if using Llama)
- ✅ MPS support available

### 5. Prepare Your Training Data

Add your email examples to `data/train_dataset.jsonl` in JSONL format:

```json
{"recipient": "john@example.com", "purpose": "Follow up", "key_points": "Point 1\nPoint 2", "tone": "professional", "email": "Your email text here..."}
```

You need at least 5-10 examples for effective fine-tuning.

### 6. Fine-tune the Model

```bash
python -m src.train
```

This will:
- Load the base model (Llama 3.2 3B)
- Apply LoRA adapters
- Fine-tune on your email examples
- Save the model to `models/lora_email_assistant/`

**Note**: Training takes ~20-40 minutes for Llama 3.2 3B with 10 examples.

## Usage

### Start the API Backend

```bash
python -m api.main
```

The API will be available at `http://localhost:8000`

### Start the Streamlit UI

In a new terminal:

```bash
streamlit run ui/streamlit_app.py
```

The UI will be available at `http://localhost:8501`

### Using the API Directly

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "recipient": "colleague@company.com",
        "purpose": "Follow up on meeting",
        "key_points": "Discuss timeline\nReview budget",
        "tone": "professional"
    }
)

print(response.json()["email"])
```

## Configuration

Edit `config.yaml` to customize:

- **Model**: Change `base_model` to use a different base LLM
- **LoRA**: Adjust `r`, `lora_alpha`, etc. for different LoRA configurations
- **Training**: Modify epochs, batch size, learning rate, etc.
- **Inference**: Toggle MPS, quantization, and other inference settings

## Model Recommendations for M3 Mac

The default model is **`meta-llama/Llama-3.2-3B-Instruct`** - best quality for email generation.

**Recommended:**
- `meta-llama/Llama-3.2-3B-Instruct` ⭐ - **Best quality** (3B params, ~6GB) - **DEFAULT**
- `gpt2` - Fast, small (124M params, ~500MB) - Good balance
- `distilgpt2` - Fastest option (82M params, ~350MB)

**Not recommended:**
- `microsoft/DialoGPT-small` - Dialogue-focused, not ideal for emails

**Note**: Llama models require Hugging Face authentication. Run `huggingface-cli login` first.

## Documentation

For more detailed information, see:

- **`GETTING_STARTED.md`** - Quick start guide with step-by-step instructions
- **`MODEL_GUIDE.md`** - Detailed model selection and configuration guide
- **`MODEL_COMPARISON.md`** - Detailed comparison between Llama 3.2 3B and GPT-2
- **`ADDING_TRAINING_DATA.md`** - Guide on how to add your training data
- **`DATA_PREP_GUIDE.md`** - Guide for preparing raw email files
- **`HUGGINGFACE_TOKEN_SETUP.md`** - Detailed Hugging Face authentication setup
- **`LLAMA_ACCESS_REQUIRED.md`** - Information about Llama model access requirements

## Tips

1. **Training Data**: More diverse examples = better generalization
2. **LoRA Rank**: Higher `r` values (16, 32) may improve quality but increase training time
3. **Quantization**: Enable for faster inference on CPU, disable for MPS
4. **Temperature**: Lower values (0.5-0.7) for more consistent outputs

## Troubleshooting

### MPS Not Available
If MPS is not available, the code will automatically fall back to CPU. Make sure you have PyTorch with MPS support.

### Out of Memory
- Reduce `per_device_train_batch_size` in `config.yaml`
- Increase `gradient_accumulation_steps`
- Use a smaller base model

### API Connection Error
Make sure the FastAPI server is running before starting the Streamlit app.

## License

This project is for personal/educational use.

