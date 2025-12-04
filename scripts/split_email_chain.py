#!/usr/bin/env python3
"""Split a long email chain into individual email examples."""
import json
import re
from pathlib import Path

def split_email_chain(input_file: str, output_file: str):
    """Split email chain by separator markers."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.loads(f.readline())
    
    email_text = data['email']
    
    # Split by separator markers
    # Look for patterns like "separator.tiff" or "---------- Forwarded message ---------"
    separators = [
        r'\nseparator\.tiff\n',
        r'\n---------- Forwarded message ---------\n',
        r'\n\nFrom: [^\n]+\nSubject: [^\n]+\nDate: [^\n]+\nTo: [^\n]+\n',
    ]
    
    # Try to split by "From:" headers that indicate new emails
    # Pattern: "From: Name <email>" followed by Subject, Date, To
    email_pattern = r'(From: [^\n]+\nSubject: [^\n]+\nDate: [^\n]+\nTo: [^\n]+(?:\nCc: [^\n]+)?(?:\nBcc: [^\n]+)?\n\n)'
    
    # Find all email starts
    matches = list(re.finditer(email_pattern, email_text))
    
    if len(matches) < 2:
        print("⚠️  Could not find multiple emails to split.")
        print("   The file may already be a single email or use a different format.")
        return
    
    print(f"📧 Found {len(matches)} potential emails in the chain")
    
    examples = []
    
    for i, match in enumerate(matches):
        start = match.start()
        # End is start of next match, or end of text
        end = matches[i + 1].start() if i + 1 < len(matches) else len(email_text)
        
        email_content = email_text[start:end].strip()
        
        # Extract recipient from "To:" line
        to_match = re.search(r'To:\s*([^\n]+)', email_content)
        recipient = to_match.group(1).strip() if to_match else data['recipient']
        
        # Extract email address from recipient
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', recipient)
        if email_match:
            recipient = email_match.group(0)
        
        # Extract subject
        subject_match = re.search(r'Subject:\s*([^\n]+)', email_content)
        subject = subject_match.group(1).strip() if subject_match else "Email"
        
        # Infer tone
        tone = "professional"  # Default
        if any(word in email_content.lower() for word in ['hey', 'hiya', 'what\'s up']):
            tone = "casual"
        elif any(word in email_content.lower() for word in ['dear sir', 'dear madam', 'respectfully']):
            tone = "formal"
        
        # Create example
        example = {
            "recipient": recipient,
            "purpose": f"PLEASE_ADD_PURPOSE - {subject[:50]}",  # Use subject as hint
            "key_points": "PLEASE_ADD_KEY_POINTS",
            "tone": tone,
            "email": email_content
        }
        
        examples.append(example)
        print(f"  ✅ Email {i+1}: {subject[:60]}...")
    
    # Save split emails
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Split into {len(examples)} individual emails → {output_file}")
    print(f"⚠️  You'll need to add purpose and key_points for each email")

if __name__ == "__main__":
    import sys
    
    input_file = "data/train_dataset.jsonl"
    output_file = "data/train_dataset.jsonl"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    split_email_chain(input_file, output_file)

