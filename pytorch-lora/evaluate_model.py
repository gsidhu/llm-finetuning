# evaluate_model.py
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
from utils import create_prompt_for_query

# --- Configuration ---
system_prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")
model_path = os.getenv("MERGED_MODEL_OUTPUT_DIRECTORY", "intent-classifier-final")
training_data_dir_path = os.getenv("TRAINING_DATA_DIRECTORY", "data")
test_data_path = f"{training_data_dir_path}/test.jsonl"

try:
    with open(system_prompt_file, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
    print(f"Successfully loaded {system_prompt_file}")
except FileNotFoundError:
    print(f"Error: {system_prompt_file} not found. Please ensure the file exists.")
    exit()

# --- 1. Load Model and Tokenizer ---
print(f"Loading model from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if torch.backends.mps.is_available():
    device_type = "mps"
    print("Using MPS (Metal Performance Shaders)")
else:
    device_type = "cpu"
    print("Using CPU")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print(f"Model loaded on device: {model.device}")

# --- 2. Load Test Dataset ---
print(f"Loading test dataset from: {test_data_path}")
test_dataset = load_dataset("json", data_files=test_data_path, split="train")
print(f"Loaded {len(test_dataset)} test examples.")

# --- 3. Prediction, Parsing, and Comparison Functions ---

def get_model_prediction(user_input):
    """
    Generates a response from the model, handling long prompts correctly.
    """
    prompt = create_prompt_for_query(user_input)
    
    # --- START OF THE FIX ---
    # Set the truncation side on the tokenizer object itself. This is the correct way.
    tokenizer.truncation_side = 'left'

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048  # Increased to handle the large system prompt
        # The 'truncation_side' argument is removed from here.
    )
    # --- END OF THE FIX ---

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")],
            pad_token_id=tokenizer.pad_token_id
        )
    
    input_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_length:]
    
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response_text.strip()


def parse_json_response(response_text):
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:-3].strip()
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, IndexError):
        return None

def compare_json_outputs(predicted_json, expected_json):
    if not predicted_json or not isinstance(expected_json, dict):
        return False

    if predicted_json.get('function') != expected_json.get('function'):
        return False

    pred_params = predicted_json.get('parameters', {})
    exp_params = expected_json.get('parameters', {})
    
    pred_params_norm = {k: str(v) for k, v in pred_params.items()}
    exp_params_norm = {k: str(v) for k, v in exp_params.items()}
    
    return pred_params_norm == exp_params_norm

# --- 4. Evaluate on Test Set ---
results = []
correct_predictions = 0
total_examples = len(test_dataset)

print(f"\nStarting evaluation on {total_examples} examples...")
for i, example in enumerate(test_dataset):
    user_input = example['User']
    expected_output_dict = example['Assistant']

    predicted_raw_text = get_model_prediction(user_input)
    predicted_json_dict = parse_json_response(predicted_raw_text)

    is_correct = compare_json_outputs(predicted_json_dict, expected_output_dict)
    
    if is_correct:
        correct_predictions += 1
    
    results.append({
        "input": user_input,
        "expected": expected_output_dict,
        "predicted_raw": predicted_raw_text,
        "predicted_parsed": predicted_json_dict,
        "correct": is_correct
    })
    
    if (i + 1) % 10 == 0:
        current_accuracy = correct_predictions / (i + 1)
        print(f"Processed {i + 1}/{total_examples} examples - Current accuracy: {current_accuracy:.4f}")

accuracy = correct_predictions / total_examples
print(f"\n--- Evaluation Complete ---")
print(f"Final Test Accuracy: {accuracy:.4f} ({correct_predictions}/{total_examples})")

# --- 5. Save and Display Results ---
output_file_path = "test_results.json"
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_examples": total_examples,
        "detailed_results": results
    }, f, indent=2, ensure_ascii=False)

print(f"Detailed results saved to {output_file_path}")

# --- Display Sample Results ---
print("\n--- Sample Predictions ---")
for i, result in enumerate(results):
    if i >= 5: break
    print(f"\n--- Example {i+1} ---")
    print(f"Input: {result['input']}")
    print(f"Expected: {json.dumps(result['expected'], indent=2, ensure_ascii=False)}")
    
    predicted_parsed = result['predicted_parsed']
    if predicted_parsed:
        confidence = predicted_parsed.get('confidence', 'N/A')
        print(f"Predicted (Parsed): {json.dumps(predicted_parsed, indent=2, ensure_ascii=False)}")
        print(f"Model Confidence: {confidence}")
    else:
        print(f"Predicted (Raw - Parse Failed): {result['predicted_raw']}")
    
    print(f"Correct: {'✅' if result['correct'] else '❌'}")
    print("-" * 80)