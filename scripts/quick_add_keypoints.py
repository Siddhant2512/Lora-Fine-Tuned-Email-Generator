#!/usr/bin/env python3
"""Quick script to add basic key_points by extracting from email content."""
import json
import re
from pathlib import Path

def extract_key_points_from_email(email_text: str, max_points: int = 3) -> str:
    """Extract potential key points from email body."""
    # Remove headers
    email_body = email_text
    for pattern in [r'From:.*?\n', r'Subject:.*?\n', r'Date:.*?\n', r'To:.*?\n', r'Cc:.*?\n']:
        email_body = re.sub(pattern, '', email_body, flags=re.IGNORECASE)
    
    # Remove forwarded message markers
    email_body = re.sub(r'---------- Forwarded message ---------.*?\n', '', email_body, flags=re.DOTALL)
    email_body = re.sub(r'separator\.tiff', '', email_body)
    
    # Find sentences that might be key points
    # Look for sentences with action words or important phrases
    sentences = re.split(r'[.!?]\s+', email_body)
    
    key_points = []
    action_words = ['request', 'please', 'kindly', 'need', 'require', 'arrange', 'inform', 'confirm']
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20 or len(sentence) > 200:
            continue
        if any(word in sentence.lower() for word in action_words):
            key_points.append(sentence)
            if len(key_points) >= max_points:
                break
    
    # If no action sentences found, use first few meaningful sentences
    if not key_points:
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if 30 < len(sentence) < 150:
                key_points.append(sentence)
                if len(key_points) >= max_points:
                    break
    
    return '\n'.join(key_points[:max_points]) if key_points else "General inquiry"

def add_key_points(input_file: str, output_file: str = None, dry_run: bool = True):
    """Add key_points to emails that are missing them."""
    if output_file is None:
        output_file = input_file
    
    examples = []
    updated = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                examples.append(example)
    
    print(f"📧 Processing {len(examples)} emails...\n")
    
    for i, example in enumerate(examples, 1):
        key_points = example.get('key_points', '').strip()
        
        # Check if key_points is missing or placeholder
        if not key_points or key_points == 'PLEASE_ADD_KEY_POINTS':
            email_text = example.get('email', '')
            extracted = extract_key_points_from_email(email_text)
            
            if not dry_run:
                example['key_points'] = extracted
            
            print(f"  Email {i}: {example.get('purpose', 'Unknown')[:50]}")
            print(f"    Old: {key_points or '(empty)'}")
            print(f"    New: {extracted[:80]}...")
            print()
            
            updated += 1
    
    if dry_run:
        print(f"⚠️  DRY RUN: Would update {updated} emails")
        print(f"   Run with --apply to actually update the file")
    else:
        # Save updated examples
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"✅ Updated {updated} emails → {output_file}")
    
    return updated

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add key_points to emails missing them")
    parser.add_argument(
        "--input",
        default="data/train_dataset.jsonl",
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file (default: overwrite input)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update the file (default: dry run)"
    )
    
    args = parser.parse_args()
    
    add_key_points(args.input, args.output, dry_run=not args.apply)

