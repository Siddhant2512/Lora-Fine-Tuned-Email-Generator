# Adding Your Training Data

## Quick Answer

**Yes!** Add your actual emails to `data/train_dataset.jsonl`

Replace the sample emails with **your own real emails** that you've actually sent.

## File Location

```
data/train_dataset.jsonl
```

## Format

Each line must be a **valid JSON object** with these fields:

```json
{"recipient": "colleague@company.com", "purpose": "Follow up on meeting", "key_points": "Point 1\nPoint 2", "tone": "professional", "email": "Your actual email text here..."}
```

## Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `recipient` | string | Who the email is for | `"colleague@company.com"` |
| `purpose` | string | Main goal of the email | `"Follow up on meeting"` |
| `key_points` | string | What to include (use `\n` for new lines) | `"Discuss timeline\nReview budget"` |
| `tone` | string | Email tone | `"professional"`, `"casual"`, `"friendly"`, `"formal"`, `"warm"` |
| `email` | string | Your actual email text | `"Hi John,\n\nI hope..."` |

## Example Entry

```json
{"recipient": "john.doe@company.com", "purpose": "Follow up on project discussion", "key_points": "Review the timeline we discussed\nConfirm next steps\nSchedule follow-up meeting", "tone": "professional", "email": "Hi John,\n\nI hope this email finds you well. I wanted to follow up on our discussion about the project timeline.\n\nAs we discussed, I've reviewed the timeline and would like to confirm the next steps. I believe we should schedule a follow-up meeting to finalize the details.\n\nPlease let me know your availability for next week.\n\nBest regards,\nSiddhant"}
```

## Important Rules

### ✅ DO:
- Use **your actual emails** that you've sent
- Include **variety**: different recipients, purposes, tones
- Use **real email text** (not templates)
- Keep JSON valid (one object per line)
- Escape special characters properly (`\n` for newlines)

### ❌ DON'T:
- Don't use placeholder text like `[Name]` or `[Your Name]`
- Don't use sample/template emails
- Don't forget commas between fields
- Don't use single quotes (use double quotes for JSON)
- Don't add trailing commas

## How Many Examples?

- **Minimum**: 5 examples
- **Recommended**: 10-15 examples
- **Optimal**: 15-20 examples
- **More is better**: Up to 30-50 for excellent results

## Tips for Best Results

### 1. Variety is Key
Include different types of emails:
- Professional emails to colleagues
- Formal emails to clients
- Casual emails to friends
- Different purposes (follow-ups, requests, updates, etc.)

### 2. Use Real Emails
- Copy emails you've actually sent
- The model learns your writing style
- Real emails = better results

### 3. Match the Format Exactly
- Each field must be present
- Use proper JSON syntax
- One email per line

### 4. Key Points Should Match Email
- The `key_points` should reflect what's actually in the email
- This helps the model learn the connection

## Step-by-Step

1. **Open** `data/train_dataset.jsonl`

2. **Delete** the sample emails (lines 1-2)

3. **Add** your emails, one per line:
   ```json
   {"recipient": "...", "purpose": "...", "key_points": "...", "tone": "...", "email": "..."}
   {"recipient": "...", "purpose": "...", "key_points": "...", "tone": "...", "email": "..."}
   ...
   ```

4. **Save** the file

5. **Verify** JSON is valid (you can test with `python -m json.tool`)

## Example: Converting Your Email

**Your Email:**
```
To: sarah@company.com
Subject: Project Update

Hi Sarah,

I wanted to give you an update on the project. We've completed phase 1 and are moving to phase 2. The timeline looks good and we're on track.

Let me know if you have any questions.

Best,
Siddhant
```

**Converted to JSON:**
```json
{"recipient": "sarah@company.com", "purpose": "Project update", "key_points": "Completed phase 1\nMoving to phase 2\nTimeline on track", "tone": "professional", "email": "Hi Sarah,\n\nI wanted to give you an update on the project. We've completed phase 1 and are moving to phase 2. The timeline looks good and we're on track.\n\nLet me know if you have any questions.\n\nBest,\nSiddhant"}
```

## Validation

After adding your emails, verify the file:

```bash
# Check if JSON is valid
python3 -c "import json; [json.loads(line) for line in open('data/train_dataset.jsonl')]"
```

If no errors, you're good to go!

## Next Steps

After adding your emails:

1. **Verify**: Run `python scripts/check_setup.py`
2. **Train**: Run `python -m src.train`
3. **Wait**: Training takes ~20-40 minutes for Llama 3.2 3B
4. **Use**: Start the API and UI to generate emails!

## Common Mistakes

### ❌ Missing Quotes
```json
{recipient: "john@email.com", ...}  // WRONG
{"recipient": "john@email.com", ...}  // CORRECT
```

### ❌ Using [Name] Placeholders
```json
{"email": "Hi [Name], ..."}  // WRONG - use actual names
{"email": "Hi John, ..."}  // CORRECT
```

### ❌ Forgetting \n for Newlines
```json
{"key_points": "Point 1 Point 2"}  // WRONG
{"key_points": "Point 1\nPoint 2"}  // CORRECT
```

### ❌ Trailing Commas
```json
{"recipient": "...", "purpose": "...",}  // WRONG
{"recipient": "...", "purpose": "..."}  // CORRECT
```

