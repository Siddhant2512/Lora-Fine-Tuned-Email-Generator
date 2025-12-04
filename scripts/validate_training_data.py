#!/usr/bin/env python3
"""Validate training data format."""
import json
import sys
from pathlib import Path

def validate_training_data(file_path: str):
    """Validate training data JSONL file."""
    data_file = Path(file_path)
    
    if not data_file.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    required_fields = ["recipient", "purpose", "key_points", "tone", "email"]
    errors = []
    examples = []
    
    print(f"Validating: {file_path}\n")
    
    with open(data_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                example = json.loads(line)
                examples.append(example)
                
                # Check required fields
                for field in required_fields:
                    if field not in example:
                        errors.append(f"Line {line_num}: Missing field '{field}'")
                
                # Check for placeholders
                if "[Name]" in example.get("email", "") or "[Your Name]" in example.get("email", ""):
                    errors.append(f"Line {line_num}: Contains placeholder [Name] or [Your Name] - use actual names")
                
                # Check tone values
                valid_tones = ["professional", "casual", "friendly", "formal", "warm"]
                if example.get("tone") not in valid_tones:
                    print(f"⚠️  Line {line_num}: Tone '{example.get('tone')}' not in recommended list: {valid_tones}")
                
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
    
    # Print results
    print(f"✅ Found {len(examples)} valid examples")
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if len(examples) < 5:
        print(f"\n⚠️  Warning: Only {len(examples)} examples. Recommended: 5-10 minimum, 10-15 for best results")
    elif len(examples) < 10:
        print(f"\n✅ Good: {len(examples)} examples. Consider adding more (10-15 optimal)")
    else:
        print(f"\n✅ Excellent: {len(examples)} examples")
    
    # Check variety
    tones = set(ex.get("tone") for ex in examples)
    recipients = set(ex.get("recipient") for ex in examples)
    
    print(f"\n📊 Variety:")
    print(f"  - Different tones: {len(tones)} ({', '.join(tones)})")
    print(f"  - Different recipients: {len(recipients)}")
    
    if len(tones) == 1:
        print(f"  ⚠️  All emails have the same tone. Add variety for better results.")
    
    print("\n✅ All examples are valid!")
    return True

if __name__ == "__main__":
    data_file = Path(__file__).parent.parent / "data" / "train_dataset.jsonl"
    success = validate_training_data(str(data_file))
    sys.exit(0 if success else 1)

