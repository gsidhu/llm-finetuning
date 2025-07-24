import json
import os

# LOAD ENV VARIABLES
from dotenv import load_dotenv
load_dotenv()

def format_prompt(example):
    """Format training examples into conversation format"""
    return f"""<|im_start|>user
{example['User']}<|im_end|>
<|im_start|>assistant
{json.dumps(example['Assistant'], ensure_ascii=False)}<|im_end|>"""

def format_prompt_for_inference(user_input, system_prompt):
    """# This format is required for the model to understand the roles correctly."""
    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""

def create_prompt_for_query(user_query: str):
    system_prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")

    # Load the system prompt from the file
    try:
        with open(system_prompt_file, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read().strip()
        print(f"Successfully loaded {system_prompt_file}")
    except FileNotFoundError:
        print(f"Error: {system_prompt_file} not found. Please ensure the file exists.")
        exit()


    # --- Construct the Full Prompt using a Chat Template ---
    final_prompt = format_prompt_for_inference(user_query, SYSTEM_PROMPT)
    
    return final_prompt