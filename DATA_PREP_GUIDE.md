# Data Preparation Guide

## Overview

The `data_prep.py` script converts your raw email `.txt` files into the training format required by the model.

## Quick Start

### Step 1: Create Raw Emails Directory

```bash
mkdir -p data/raw_emails
```

### Step 2: Add Your Email Files

Copy your emails as `.txt` files to `data/raw_emails/`

### Step 3: Create Metadata (Recommended)

```bash
python src/data_prep.py --create-templates
```

This creates `.meta.json` files for each email. Edit them to add:
- `recipient`: Who the email is for
- `purpose`: What the email is about
- `key_points`: Main points (use `\n` for new lines)
- `tone`: professional, casual, friendly, formal, or warm

### Step 4: Process Emails

```bash
python src/data_prep.py --raw-dir data/raw_emails --output data/train_dataset.jsonl
```

This creates/updates `data/train_dataset.jsonl` with your emails in the correct format.

## Two Methods

### Method 1: With Metadata Files (Best)

**Advantages:**
- ✅ Organized
- ✅ Easy to edit
- ✅ Can reuse metadata
- ✅ Clear separation of concerns

**Steps:**
1. Add `.txt` email files
2. Run `--create-templates` to generate `.meta.json` files
3. Edit `.meta.json` files with recipient, purpose, key_points, tone
4. Run processing script

**Example:**
```
data/raw_emails/
├── email1.txt
├── email1.meta.json
├── email2.txt
└── email2.meta.json
```

### Method 2: Direct Processing (Faster)

**Advantages:**
- ✅ Quick for testing
- ✅ Good if you have few emails

**Steps:**
1. Add `.txt` email files
2. Run processing script
3. Edit the output JSONL file directly to add missing fields

**Note:** Script will try to infer recipient and tone, but you'll need to manually add purpose and key_points.

## Command Line Options

### Basic Processing
```bash
python src/data_prep.py
```

### Custom Directories
```bash
python src/data_prep.py --raw-dir path/to/emails --output path/to/output.jsonl
```

### Create Templates
```bash
python src/data_prep.py --create-templates
```

### Anonymize Sensitive Data
```bash
python src/data_prep.py --remove-sensitive
```
This will:
- Replace email addresses with `[EMAIL]`
- Replace phone numbers with `[PHONE]`
- Replace URLs with `[URL]`

### Append to Existing File
```bash
python src/data_prep.py --append
```
Useful when adding more emails to an existing dataset.

## Metadata File Format

Each `.meta.json` file should contain:

```json
{
  "recipient": "colleague@company.com",
  "purpose": "Follow up on meeting",
  "key_points": "Point 1\nPoint 2\nPoint 3",
  "tone": "professional"
}
```

### Tone Options
- `"professional"` - Business emails
- `"casual"` - Informal emails
- `"friendly"` - Warm but professional
- `"formal"` - Very formal
- `"warm"` - Personal and warm

## Email File Format

Your `.txt` files can include:

**Option 1: Plain email body**
```
Hi John,

I wanted to follow up on our meeting...

Best,
Siddhant
```

**Option 2: With headers**
```
To: john@company.com
Subject: Follow up

Hi John,

I wanted to follow up on our meeting...

Best,
Siddhant
```

The script will try to extract the recipient from headers if present.

## Workflow Example

### Complete Workflow

1. **Prepare emails:**
   ```bash
   # Copy your emails to raw_emails/
   cp my_emails/*.txt data/raw_emails/
   ```

2. **Create templates:**
   ```bash
   python src/data_prep.py --create-templates
   ```

3. **Edit metadata files:**
   - Open each `.meta.json` file
   - Fill in recipient, purpose, key_points, tone

4. **Process:**
   ```bash
   python src/data_prep.py
   ```

5. **Validate:**
   ```bash
   python scripts/validate_training_data.py
   ```

6. **Train:**
   ```bash
   python -m src.train
   ```

## Tips

### Best Practices

1. **Use real emails**: Copy emails you've actually sent
2. **Variety**: Include different tones, recipients, purposes
3. **Quality over quantity**: 10 good examples > 20 poor ones
4. **Match metadata**: Ensure key_points reflect what's in the email

### Common Issues

**Problem**: "PLEASE_ADD_PURPOSE" in output
- **Solution**: Add metadata files or edit JSONL directly

**Problem**: Recipient not detected
- **Solution**: Add recipient to metadata file or email header

**Problem**: Wrong tone detected
- **Solution**: Override in metadata file

## Output Format

The script creates entries in this format:

```json
{"recipient": "...", "purpose": "...", "key_points": "...", "tone": "...", "email": "..."}
```

This matches exactly what the training script expects.

## Next Steps

After processing:

1. **Validate**: `python scripts/validate_training_data.py`
2. **Review**: Check the output JSONL file
3. **Train**: `python -m src.train`

