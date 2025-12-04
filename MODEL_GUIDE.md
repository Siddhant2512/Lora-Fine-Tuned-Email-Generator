# Model Selection Guide

## Current Default: `meta-llama/Llama-3.2-3B-Instruct` ⭐

**Why Llama 3.2 3B?**
- ✅ Best quality for email generation
- ✅ Excellent instruction following
- ✅ Modern architecture with chat template support
- ✅ Professional, coherent email outputs
- ✅ Works well with LoRA fine-tuning
- ✅ MPS acceleration on M3 Mac

## Model Comparison

### Recommended Models

| Model | Parameters | Size | Speed | Quality | Best For |
|-------|-----------|------|-------|---------|----------|
| **`meta-llama/Llama-3.2-3B-Instruct`** ⭐ | 3B | ~6GB | Medium | Excellent | **Default - Best quality** |
| `gpt2` | 124M | ~500MB | Fast | Good | Fast iteration |
| `distilgpt2` | 82M | ~350MB | Fastest | Good | Maximum speed |
| `gpt2-medium` | 355M | ~1.4GB | Medium | Better | Better quality than GPT-2 |

### Not Recommended

| Model | Why Not |
|-------|---------|
| `microsoft/DialoGPT-small` | Dialogue-focused, not ideal for email structure |
| `gpt2-large` | Too large for M3 Mac (774M params) |
| `gpt2-xl` | Too large for M3 Mac (1.5B params) |

## How to Change Models

Edit `config.yaml`:

```yaml
model:
  base_model: "meta-llama/Llama-3.2-3B-Instruct"  # Default
  # OR
  base_model: "gpt2"  # For faster training/testing
```

### For Different Model Types

**Llama Models** (recommended):
- `meta-llama/Llama-3.2-3B-Instruct` - Default, best quality ⭐
- Auto-detects target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]`

**GPT-2 Models** (faster alternatives):
- `gpt2` - Fast, good quality
- `distilgpt2` - Fastest option
- `gpt2-medium` - Better quality than GPT-2
- Auto-detects target modules: `["c_attn", "c_proj"]`

**OPT Models** (alternative):
- `facebook/opt-125m` - Similar size to GPT-2
- Auto-detects target modules: `["q_proj", "v_proj", "k_proj", "out_proj"]`

## Model Selection Tips

1. **Start with Llama 3.2 3B** (default) - Best quality for production
2. **Need speed for testing?** → Use `gpt2` temporarily
3. **Limited RAM (8GB)?** → Use `gpt2` or `distilgpt2`
4. **Training time**: Larger models = longer training, but LoRA keeps it manageable

## Performance on M3 Mac

### Llama 3.2 3B (Default)
- **Inference**: ~5-15 seconds per email
- **Training**: ~20-40 minutes (5 epochs, 10 examples)
- **Memory**: 8-12 GB RAM
- **Quality**: ⭐⭐⭐⭐⭐ Excellent

### GPT-2 (Alternative)
- **Inference**: ~2-5 seconds per email
- **Training**: ~5-10 minutes (5 epochs, 10 examples)
- **Memory**: 2-4 GB RAM
- **Quality**: ⭐⭐⭐ Good

### DistilGPT-2 (Fastest)
- **Inference**: ~1-3 seconds per email
- **Training**: ~3-7 minutes (5 epochs, 10 examples)
- **Memory**: 2-3 GB RAM
- **Quality**: ⭐⭐⭐ Good

## After Changing Models

1. Delete old model: `rm -rf models/lora_email_assistant/`
2. Retrain: `python -m src.train`
3. Restart API: `python -m api.main`

## Authentication Requirements

**Llama Models** (including default):
- ✅ Requires Hugging Face authentication
- Run: `huggingface-cli login`
- Get token: https://huggingface.co/settings/tokens

**GPT-2 Models**:
- ✅ No authentication needed
- Works immediately after installation

## Why Llama 3.2 3B?

Llama 3.2 3B-Instruct is specifically designed for:
- ✅ Instruction following (perfect for structured emails)
- ✅ Professional text generation
- ✅ Chat template support (better prompt handling)
- ✅ Modern architecture (better than GPT-2)
- ✅ Fine-tuned for helpful, safe outputs

GPT-2 is older and:
- ⚠️ Less instruction following
- ⚠️ May need more fine-tuning
- ⚠️ Lower quality outputs
- ✅ But faster and simpler

## Recommendation

**For Production**: Use **Llama 3.2 3B** (default) ⭐
- Best quality is worth the extra time
- Better instruction following
- More professional emails

**For Quick Testing**: Use **GPT-2** temporarily
- Fast iteration
- No authentication needed
- Good for experimenting
