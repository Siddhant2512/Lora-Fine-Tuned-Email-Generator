"""
Data preparation script for converting raw emails to training format.

Supports two modes:
1. With metadata file: Each email has a corresponding .meta.json file
2. Manual mode: Creates template files for manual annotation
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Optional, List


def clean_email(text: str, remove_sensitive: bool = False) -> str:
    """
    Clean email text. By default, keeps original formatting.
    Set remove_sensitive=True to anonymize data.
    """
    if not remove_sensitive:
        # Just normalize whitespace (preserve line breaks)
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r' *\n *', '\n', text)  # Clean line breaks
        return text.strip()
    
    # Full cleaning mode (for anonymization)
    # Remove email addresses (but keep recipient field separate)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Remove phone numbers (US format)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def parse_rtf(rtf_text: str) -> str:
    """
    Extract plain text from RTF format.
    Improved RTF parser that handles common RTF structures.
    """
    try:
        # Try using striprtf library if available
        from striprtf.striprtf import rtf_to_text
        text = rtf_to_text(rtf_text)
        return text
    except ImportError:
        # Fallback to basic parser
        pass
    
    # Basic RTF parser (fallback)
    text = rtf_text
    
    # Remove RTF header and font tables (everything before first \par or content)
    text = re.sub(r'^.*?\\pard', '', text, flags=re.DOTALL)
    
    # Replace RTF line breaks
    text = text.replace('\\par', '\n')
    text = text.replace('\\line', '\n')
    text = text.replace('\\tab', '\t')
    text = text.replace('\\pard', '\n')
    
    # Remove RTF control words (but keep text)
    text = re.sub(r'\\[a-z]+\d*\s*', ' ', text)
    text = re.sub(r'\\[^a-z\s]', '', text)
    
    # Remove RTF groups (but extract text from them)
    # This is a simplified approach - extract text between braces
    def extract_text_from_braces(match):
        content = match.group(1)
        # Remove control codes, keep text
        content = re.sub(r'\\[a-z]+\d*\s*', ' ', content)
        return content
    
    # Remove nested braces but keep their content
    while '{' in text:
        text = re.sub(r'\{([^{}]*)\}', r'\1', text)
    
    # Decode Unicode escapes like \u8239
    def decode_unicode(match):
        code = int(match.group(1))
        try:
            return chr(code)
        except:
            return ''
    
    text = re.sub(r'\\u(\d+)', decode_unicode, text)
    
    # Remove remaining RTF codes
    text = re.sub(r'\\[a-z]+\d*\s*', '', text)
    text = re.sub(r'\\[^a-zA-Z]', '', text)
    
    # Clean up special characters
    text = text.replace("\\'a0", ' ')  # Non-breaking space
    text = text.replace("\\'", "'")  # Apostrophes
    
    # Extract email addresses from HYPERLINK fields
    email_pattern = r'mailto:([^\s"{}]+)'
    emails = re.findall(email_pattern, text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines
    text = text.strip()
    
    return text


def anonymize_names(text: str, name_mapping: Optional[Dict[str, str]] = None) -> str:
    """Replace real names with placeholders."""
    if name_mapping is None:
        name_mapping = {}
    
    for real_name, placeholder in name_mapping.items():
        text = text.replace(real_name, placeholder)
    
    return text


def infer_tone_from_email(email_text: str) -> str:
    """
    Attempt to infer tone from email content.
    Returns a guess, but should be manually verified.
    """
    email_lower = email_text.lower()
    
    # Casual indicators
    if any(word in email_lower for word in ['hey', 'hiya', 'what\'s up', 'talk soon', 'catch you later']):
        return "casual"
    
    # Formal indicators
    if any(word in email_lower for word in ['dear sir', 'dear madam', 'respectfully', 'yours sincerely']):
        return "formal"
    
    # Friendly indicators
    if any(word in email_lower for word in ['hope you\'re well', 'great to hear', 'looking forward']):
        return "friendly"
    
    # Warm indicators
    if any(word in email_lower for word in ['warm regards', 'best wishes', 'take care']):
        return "warm"
    
    # Default to professional
    return "professional"


def extract_recipient_from_email(email_text: str) -> Optional[str]:
    """Try to extract recipient from email headers or content."""
    # Check for common email header patterns
    patterns = [
        r'To:\s*([^\n]+)',
        r'Recipient:\s*([^\n]+)',
        r'Dear\s+([^,\n]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, email_text, re.IGNORECASE)
        if match:
            recipient = match.group(1).strip()
            # Extract email if present
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', recipient)
            if email_match:
                return email_match.group(0)
            # Or return name if no email
            return recipient
    
    return None


def load_metadata(metadata_file: Path) -> Optional[Dict]:
    """Load metadata from JSON file."""
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Warning: Could not load metadata from {metadata_file}: {e}")
        return None


def process_email_file(
    email_file: Path,
    remove_sensitive: bool = False,
    name_mapping: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Process a single email file into training format.
    
    Format: {"recipient": "...", "purpose": "...", "key_points": "...", "tone": "...", "email": "..."}
    """
    # Read email text
    with open(email_file, 'r', encoding='utf-8', errors='ignore') as f:
        email_text = f.read()
    
    # Handle RTF files
    if email_file.suffix.lower() == '.rtf':
        email_text = parse_rtf(email_text)
    
    # Clean email (optional)
    cleaned_email = clean_email(email_text, remove_sensitive=remove_sensitive)
    if name_mapping:
        cleaned_email = anonymize_names(cleaned_email, name_mapping)
    
    # Try to load metadata file (same name but .meta.json extension)
    metadata_file = email_file.with_suffix('.meta.json')
    metadata = load_metadata(metadata_file)
    
    # Extract or use metadata
    recipient = None
    purpose = None
    key_points = None
    tone = None
    
    if metadata:
        # Use metadata if available
        recipient = metadata.get('recipient')
        purpose = metadata.get('purpose')
        key_points = metadata.get('key_points')
        tone = metadata.get('tone')
    else:
        # Try to infer from email
        recipient = extract_recipient_from_email(email_text) or "unknown@example.com"
        tone = infer_tone_from_email(email_text)
        # Purpose and key_points must be provided manually
    
    # Create example in our format
    example = {
        "recipient": recipient or "unknown@example.com",
        "purpose": purpose or "PLEASE_ADD_PURPOSE",
        "key_points": key_points or "PLEASE_ADD_KEY_POINTS",
        "tone": tone or "professional",
        "email": cleaned_email
    }
    
    return example


def process_raw_emails(
    raw_dir: str = "data/raw_emails",
    output_file: str = "data/train_dataset.jsonl",
    remove_sensitive: bool = False,
    name_mapping: Optional[Dict[str, str]] = None,
    append: bool = False
) -> None:
    """
    Process all raw email .txt files into JSONL format.
    
    Args:
        raw_dir: Directory containing .txt email files
        output_file: Output JSONL file path
        remove_sensitive: If True, anonymize emails
        name_mapping: Dict mapping real names to placeholders
        append: If True, append to existing file; otherwise overwrite
    """
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        print(f"❌ Directory not found: {raw_dir}")
        print(f"   Create the directory and add your .txt email files there.")
        return
    
    # Find all email files (.txt, .rtf, and .rtf in subdirectories)
    email_files = []
    email_files.extend(raw_path.glob("*.txt"))
    email_files.extend(raw_path.glob("*.rtf"))
    email_files.extend(raw_path.rglob("*.rtf"))  # Also search in subdirectories
    
    # Remove duplicates
    email_files = list(set(email_files))
    
    if not email_files:
        print(f"⚠️  No email files (.txt or .rtf) found in {raw_dir}")
        return
    
    print(f"📧 Processing {len(email_files)} email files...\n")
    
    examples = []
    needs_annotation = []
    
    for email_file in sorted(email_files):
        try:
            example = process_email_file(email_file, remove_sensitive, name_mapping)
            
            # Check if needs manual annotation
            if example["purpose"] == "PLEASE_ADD_PURPOSE" or example["key_points"] == "PLEASE_ADD_KEY_POINTS":
                needs_annotation.append(email_file.name)
            
            examples.append(example)
            print(f"  ✅ {email_file.name}")
            
        except Exception as e:
            print(f"  ❌ {email_file.name}: {e}")
    
    # Save to JSONL
    mode = 'a' if append else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Processed {len(examples)} emails → {output_file}")
    
    if needs_annotation:
        print(f"\n⚠️  {len(needs_annotation)} emails need manual annotation:")
        print("   Add .meta.json files or edit the JSONL file directly.")
        print("   Files needing annotation:")
        for filename in needs_annotation:
            print(f"     - {filename}")


def create_metadata_template(email_file: Path) -> None:
    """Create a metadata template file for an email."""
    metadata_file = email_file.with_suffix('.meta.json')
    
    template = {
        "recipient": "recipient@example.com",
        "purpose": "Brief purpose of the email",
        "key_points": "Key point 1\nKey point 2\nKey point 3",
        "tone": "professional"  # Options: professional, casual, friendly, formal, warm
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Created metadata template: {metadata_file}")


def create_metadata_templates(raw_dir: str = "data/raw_emails") -> None:
    """Create metadata templates for all email files."""
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        print(f"❌ Directory not found: {raw_dir}")
        return
    
    email_files = list(raw_path.glob("*.txt"))
    
    if not email_files:
        print(f"⚠️  No .txt files found in {raw_dir}")
        return
    
    print(f"📝 Creating metadata templates for {len(email_files)} files...\n")
    
    for email_file in sorted(email_files):
        metadata_file = email_file.with_suffix('.meta.json')
        if not metadata_file.exists():
            create_metadata_template(email_file)
        else:
            print(f"  ⏭️  {email_file.name} (metadata already exists)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process raw emails into training format")
    parser.add_argument(
        "--raw-dir",
        default="data/raw_emails",
        help="Directory containing raw .txt email files"
    )
    parser.add_argument(
        "--output",
        default="data/train_dataset.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--remove-sensitive",
        action="store_true",
        help="Remove/anonymize sensitive information"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file instead of overwriting"
    )
    parser.add_argument(
        "--create-templates",
        action="store_true",
        help="Create metadata template files for manual annotation"
    )
    
    args = parser.parse_args()
    
    if args.create_templates:
        create_metadata_templates(args.raw_dir)
    else:
        process_raw_emails(
            raw_dir=args.raw_dir,
            output_file=args.output,
            remove_sensitive=args.remove_sensitive,
            append=args.append
        )
