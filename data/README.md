# Training Data

## Important: Sensitive Data Protection

The `train_dataset.jsonl` file contains real email addresses and personal information, so it is **not included** in this repository for privacy and security reasons.

## Setting Up Your Training Data

### Option 1: Create Your Own Training File

Create a file named `train_dataset.jsonl` in this directory with your email examples.

**Format:** Each line should be a JSON object with the following structure:

```json
{"recipient": "recipient@example.com", "purpose": "Brief purpose description", "key_points": "Key point 1\nKey point 2", "tone": "professional", "email": "Your complete email text here..."}
```

**Required fields:**
- `recipient`: Email recipient address
- `purpose`: Main goal/purpose of the email
- `key_points`: Key points to include (use `\n` for new lines)
- `tone`: Email tone (`professional`, `casual`, `friendly`, `formal`, `warm`)
- `email`: Complete email text

### Option 2: Use Sample Examples

See `examples/sample_examples.jsonl` for example format and structure.

### Option 3: Prepare from Raw Emails

If you have raw email files, use the data preparation script:

```bash
python src/data_prep.py
```

See `DATA_PREP_GUIDE.md` in the root directory for detailed instructions.

## Minimum Requirements

- **At least 5-10 examples** for basic fine-tuning
- **20+ examples** for better quality and generalization
- **Diverse examples** covering different tones, purposes, and recipients

## Privacy Note

Your `train_dataset.jsonl` file is automatically ignored by git (see `.gitignore`), so it will never be accidentally committed to the repository.

