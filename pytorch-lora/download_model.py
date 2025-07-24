# download_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# LOAD ENV VARIABLES
from dotenv import load_dotenv
load_dotenv()

# Read model name from environment variable
import os
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
print(f"Downloading model: {model_name}")

# Download model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # Changed from "mps" to "auto" for better compatibility
)

# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model downloaded: {model_name}")
print(f"Model parameters: {model.num_parameters():,}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")