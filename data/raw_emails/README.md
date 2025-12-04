# Raw Emails Directory

Place your raw email `.txt` files here. The `data_prep.py` script will convert them to the training format.

## Two Ways to Use

### Method 1: With Metadata Files (Recommended)

1. **Add your email files** as `.txt` files
2. **Create corresponding `.meta.json` files** with the same name

Example:
- `email1.txt` - Your email text
- `email1.meta.json` - Metadata

**Metadata format** (`email1.meta.json`):
```json
{
  "recipient": "colleague@company.com",
  "purpose": "Follow up on meeting",
  "key_points": "Discuss project timeline\nReview budget\nSchedule next steps",
  "tone": "professional"
}
```

### Method 2: Auto-Process (Then Edit)

1. **Add your email files** as `.txt` files
2. **Run the script** - it will try to infer recipient and tone
3. **Edit the output** JSONL file to add purpose and key_points

## Usage

### Create metadata templates:
```bash
python src/data_prep.py --create-templates
```

### Process emails:
```bash
python src/data_prep.py --raw-dir data/raw_emails --output data/train_dataset.jsonl
```

### Process with anonymization:
```bash
python src/data_prep.py --remove-sensitive
```

### Append to existing file:
```bash
python src/data_prep.py --append
```

## Email File Format

Your `.txt` files can be:
- Plain email text
- Email with headers (To:, Subject:, etc.)
- Just the email body

The script will try to extract recipient from headers if present.

## Example Email File

`email1.txt`:
```
To: john@company.com
Subject: Project Update

Hi John,

I wanted to give you an update on the project. We've completed phase 1 and are moving to phase 2.

Let me know if you have any questions.

Best,
Siddhant
```

## Example Metadata File

`email1.meta.json`:
```json
{
  "recipient": "john@company.com",
  "purpose": "Project update",
  "key_points": "Completed phase 1\nMoving to phase 2\nTimeline on track",
  "tone": "professional"
}
```

