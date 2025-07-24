# merge_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# --- Configuration ---
base_model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
adapter_path = os.getenv("FINETUNED_MODEL_OUTPUT_DIRECTORY", "finetuned-model")
output_dir = os.getenv("MERGED_MODEL_OUTPUT_DIRECTORY", "merged-model")

print("--- Step 1: Loading Base Model and Tokenizer ---")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the base model
# IMPROVEMENT: Load the model on the CPU to avoid memory issues on MPS during merge.
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,  # Use float16 to save memory
    device_map="cpu",           # Explicitly load to CPU
    low_cpu_mem_usage=True
)
print(f"Base model '{base_model_name}' loaded on CPU.")

print("\n--- Step 2: Loading and Merging LoRA Adapter ---")
# Load the PEFT model by applying the adapter to the base model
model = PeftModel.from_pretrained(base_model, adapter_path)
print(f"LoRA adapter from '{adapter_path}' loaded.")

# Merge the adapter weights into the base model
# This returns a new, standalone model
merged_model = model.merge_and_unload()
print("LoRA weights merged successfully.")

print(f"\n--- Step 3: Saving the Merged Model ---")
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the merged model and the tokenizer
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Final merged model has been saved to: {output_dir}")