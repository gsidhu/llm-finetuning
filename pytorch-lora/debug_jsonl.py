# debug_jsonl.py
# Diagnose and fix JSONL file issues

import json
import os
from pathlib import Path

def diagnose_jsonl_file(file_path):
    """Diagnose issues with JSONL file"""
    print(f"Diagnosing file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"📁 File size: {file_size} bytes")
    
    if file_size == 0:
        print("❌ File is empty!")
        return False
    
    # Read raw bytes to check for encoding issues
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
        print(f"🔍 Raw file length: {len(raw_content)} bytes")
        
        # Check for BOM (Byte Order Mark)
        if raw_content.startswith(b'\xef\xbb\xbf'):
            print("⚠️  File has UTF-8 BOM (Byte Order Mark)")
        
        # Check for null bytes
        if b'\x00' in raw_content:
            print("⚠️  File contains null bytes")
            
    except Exception as e:
        print(f"❌ Error reading raw file: {e}")
        return False
    
    # Try to read as text
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"📄 Text content length: {len(content)} characters")
        
        # Show first few characters with repr to see hidden characters
        preview = repr(content[:200])
        print(f"🔍 Content preview: {preview}")
        
    except UnicodeDecodeError as e:
        print(f"❌ UTF-8 encoding error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading text file: {e}")
        return False
    
    # Try to parse each line as JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📋 Number of lines: {len(lines)}")
        
        valid_lines = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                print(f"⚠️  Line {i+1} is empty")
                continue
                
            try:
                json.loads(line)
                valid_lines += 1
            except json.JSONDecodeError as e:
                print(f"❌ Line {i+1} JSON error: {e}")
                print(f"   Content: {repr(line)}")
                return False
        
        print(f"✅ All {valid_lines} non-empty lines are valid JSON")
        return True
        
    except Exception as e:
        print(f"❌ Error processing lines: {e}")
        return False

def clean_jsonl_file(input_path, output_path=None):
    """Clean and fix JSONL file"""
    if output_path is None:
        output_path = input_path.replace('.jsonl', '_cleaned.jsonl')
    
    print(f"🧹 Cleaning {input_path} -> {output_path}")
    
    try:
        # Read the file and clean it
        with open(input_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
            lines = f.readlines()
        
        cleaned_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                # Parse and re-serialize to ensure proper formatting
                data = json.loads(line)
                cleaned_line = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
                cleaned_lines.append(cleaned_line)
            except json.JSONDecodeError as e:
                print(f"❌ Skipping invalid line {i+1}: {e}")
                continue
        
        # Write cleaned file
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in cleaned_lines:
                f.write(line + '\n')
        
        print(f"✅ Cleaned file saved with {len(cleaned_lines)} valid lines")
        return output_path
        
    except Exception as e:
        print(f"❌ Error cleaning file: {e}")
        return None

def test_with_datasets_library(file_path):
    """Test loading with datasets library"""
    try:
        from datasets import load_dataset
        print(f"🧪 Testing with datasets library: {file_path}")
        
        dataset = load_dataset("json", data_files=file_path, split="train")
        print(f"✅ Successfully loaded {len(dataset)} examples")
        
        # Show first example
        if len(dataset) > 0:
            print(f"📋 First example: {dataset[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Datasets library error: {e}")
        return False

if __name__ == "__main__":
    # Test your files
    training_data_directory = os.getenv("TRAINING_DATA_DIRECTORY", "data")
    jsonl_files = Path(training_data_directory).glob("*.jsonl")
    files_to_check = [str(file) for file in jsonl_files if file.is_file()]
    
    for file_path in files_to_check:
        print("="*60)
        if diagnose_jsonl_file(file_path):
            # Try loading with datasets
            if not test_with_datasets_library(file_path):
                # If datasets fails, try cleaning
                cleaned_path = clean_jsonl_file(file_path)
                if cleaned_path:
                    test_with_datasets_library(cleaned_path)
        print()