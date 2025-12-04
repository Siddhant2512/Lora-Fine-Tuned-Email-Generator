# Email Examples

Place your email examples here in JSONL format (one JSON object per line).

## Format

Each line should be a JSON object with the following structure:

```json
{"recipient": "john@example.com", "purpose": "Follow up on meeting", "key_points": "Discuss project timeline\nReview budget", "tone": "professional", "email": "Dear John,\n\nI hope this email finds you well. I wanted to follow up on our meeting yesterday..."}
```

## Fields

- `recipient`: Email recipient (string)
- `purpose`: Purpose of the email (string)
- `key_points`: Key points to include (string, can be multiline)
- `tone`: Tone of the email (string: "professional", "casual", "friendly", "formal", "warm")
- `email`: The actual email text (string)

## Example File

See `sample_examples.jsonl` for a template.

