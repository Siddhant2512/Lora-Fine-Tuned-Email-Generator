# Getting Started with LoRA-Mail Assistant (Llama 3.2 3B)

## Quick Start (5 Steps)

### Step 1: Install Dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Authenticate with Hugging Face

Llama models require authentication. First, create a token:

1. **Create Token:**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - **Token Name**: `lora-mail-assistant` (or any name)
   - **Token Type**: Select **"Read"** (this is enough for downloading models)
   - Click "Generate token"
   - **IMPORTANT**: Copy the token immediately (you won't see it again!)

2. **Login with token:**
   ```bash
   huggingface-cli login
   ```
   Paste your token when prompted.

**📖 Detailed instructions**: See `HUGGINGFACE_TOKEN_SETUP.md` for complete token creation guide with all details.

### Step 3: Verify Setup

```bash
python scripts/check_setup.py
```

Fix any issues it reports.

### Step 4: Add Your Email Examples

Edit `data/train_dataset.jsonl` and add 5-10 of your actual emails:

```json
{"recipient": "colleague@company.com", "purpose": "Follow up on meeting", "key_points": "Discuss timeline\nReview budget", "tone": "professional", "email": "Hi John,\n\nI hope this email finds you well. I wanted to follow up on our meeting yesterday about the project timeline.\n\nAs we discussed, I've reviewed the timeline and would like to confirm the next steps. I believe we should schedule a follow-up meeting to finalize the details.\n\nPlease let me know your availability for next week.\n\nBest regards,\n[Your Name]"}
```

**Important**: Use YOUR actual emails to train the model in YOUR writing style!

### Step 5: Train the Model

```bash
python -m src.train
```

This will:
- Download Llama 3.2 3B (~6GB, first time only)
- Fine-tune with LoRA on your examples
- Save to `models/lora_email_assistant/`

**Time**: ~20-40 minutes for 10 examples on M3 Mac

## After Training

### Start the API

```bash
python -m api.main
```

API runs on: http://localhost:8000

### Start the UI (in another terminal)

```bash
streamlit run ui/streamlit_app.py
```

UI opens at: http://localhost:8501

## Using the Assistant

1. Open the Streamlit UI
2. Fill in:
   - **Recipient**: Who the email is for
   - **Purpose**: Main goal of the email
   - **Key Points**: What to include
   - **Tone**: professional, casual, friendly, etc.
3. Click "Generate Email"
4. Review and download the result

## Tips for Best Results

### Training Data Quality

- ✅ Use **real emails** you've actually sent
- ✅ Include **variety**: different recipients, purposes, tones
- ✅ **5-10 examples minimum**, 15-20 is better
- ✅ Match the format exactly (recipient, purpose, key_points, tone, email)

### Model Configuration

- **More examples** = better results
- **Higher LoRA rank** (r=16) = more capacity but slower
- **More epochs** (10-15) = better learning but longer training

### Generation Tips

- Be **specific** in "Purpose" and "Key Points"
- Use **clear, descriptive** key points
- The model learns from your examples, so **quality training data matters most**

## Troubleshooting

### "Authentication required"
```bash
huggingface-cli login
```

### "Out of memory"
- Reduce batch size in `config.yaml`: `per_device_train_batch_size: 1`
- Close other applications
- Use GPT-2 instead (smaller model)

### "Model not found"
- Check internet connection (first download is ~6GB)
- Verify authentication: `huggingface-cli whoami`

### Slow generation
- Normal for Llama 3.2 3B: 5-15 seconds per email
- Ensure MPS is enabled (check in UI sidebar)
- Disable quantization if using MPS: `use_quantization: false` in config.yaml

## Next Steps

- Add more training examples for better results
- Experiment with different tones and styles
- Fine-tune the prompt format in `src/model_loader.py`
- Adjust generation parameters in `config.yaml`

## Need Help?

- Check `README.md` for full documentation
- See `MODEL_COMPARISON.md` for model details
- Review `README.md` for full documentation

