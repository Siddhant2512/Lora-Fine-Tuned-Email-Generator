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

## MPS (Metal Performance Shaders) Acceleration

This project is optimized for Apple Silicon Macs (M1, M2, M3) using **Metal Performance Shaders (MPS)** for GPU acceleration. MPS is Apple's framework that allows PyTorch to leverage the GPU cores in Apple Silicon chips for neural network computations.

### What is MPS?

MPS provides a unified API for GPU-accelerated machine learning on macOS. Instead of using CUDA (NVIDIA GPUs) or running on CPU, MPS enables PyTorch to execute tensor operations directly on Apple's Metal GPU, resulting in significantly faster inference and training.

### Performance Benefits

- **3-5x faster inference** compared to CPU-only execution
- **Lower memory usage** through efficient GPU memory management
- **Native integration** with PyTorch - no additional drivers needed
- **Automatic fallback** to CPU if MPS is unavailable

### How It Works in This Project

The code automatically detects MPS availability and uses it for:
- Model loading and inference (email generation)
- LoRA fine-tuning (if supported)
- Tensor operations during forward passes

MPS is enabled by default in `config.yaml`:
```yaml
inference:
  use_mps: true
  device: "mps"  # Automatically falls back to "cpu" if MPS unavailable
```

### Requirements

- **macOS 12.3+** (Monterey or later)
- **Apple Silicon Mac** (M1, M2, M3, or later)
- **PyTorch 1.12+** with MPS support (included in requirements.txt)

### Verifying MPS Support

Check if MPS is available on your system:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

Or use the setup verification script:
```bash
python scripts/check_setup.py
```

### Performance Notes

- **First generation**: ~8-12 seconds (includes model loading)
- **Subsequent generations**: ~8-12 seconds per email (with Llama 3.2 3B)
- **Model compilation**: The code attempts to use `torch.compile()` for additional 20-30% speedup when available
- **Memory**: Llama 3.2 3B requires ~8-12GB RAM with MPS acceleration

### Limitations

- **Quantization**: 8-bit quantization (BitsAndBytes) doesn't work with MPS, so the model uses float16 precision instead
- **Training**: Some training operations may still use CPU, but inference is fully GPU-accelerated
- **CUDA models**: Models trained on CUDA can be used with MPS without modification

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
- **`ADDING_TRAINING_DATA.md`** - Guide on how to add your training data
- **`DATA_PREP_GUIDE.md`** - Guide for preparing raw email files
- **`HUGGINGFACE_TOKEN_SETUP.md`** - Detailed Hugging Face authentication setup
- **`LLAMA_ACCESS_REQUIRED.md`** - Information about Llama model access requirements

## Tips

1. **Training Data**: More diverse examples = better generalization
2. **LoRA Rank**: Higher `r` values (16, 32) may improve quality but increase training time
3. **Quantization**: Not used with MPS (automatically disabled). For CPU inference, enable in config.yaml
4. **Temperature**: Lower values (0.5-0.7) for more consistent outputs
5. **MPS Optimization**: Keep other applications closed during inference for best GPU performance
6. **Memory Management**: The code automatically clears MPS cache between generations to prevent memory buildup

## Troubleshooting

### MPS Not Available
If MPS is not available, the code will automatically fall back to CPU. Common causes:
- **Intel Mac**: MPS only works on Apple Silicon (M1/M2/M3). Intel Macs will use CPU.
- **Old macOS**: Requires macOS 12.3+ (Monterey or later)
- **PyTorch version**: Ensure you have PyTorch 1.12+ with MPS support

To check MPS availability:
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

**Note**: CPU inference will work but will be slower (~20-30 seconds per email with Llama 3.2 3B).

### Out of Memory
- Reduce `per_device_train_batch_size` in `config.yaml`
- Increase `gradient_accumulation_steps`
- Use a smaller base model

### API Connection Error
Make sure the FastAPI server is running before starting the Streamlit app.

## License

This project is for personal/educational use.

