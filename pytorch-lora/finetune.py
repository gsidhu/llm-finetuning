# finetune_with_checkpointing.py
import torch
import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
from utils import format_prompt

# LOAD ENV VARIABLES
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
# Read model name from environment variable
import os
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
print(f"Using model: {model_name}")
output_dir = os.getenv("FINETUNED_MODEL_OUTPUT_DIRECTORY", "finetuned-model")
resume_from_checkpoint = None  # Set to checkpoint path to resume

# Check for existing checkpoints
if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        # Get the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        resume_from_checkpoint = os.path.join(output_dir, latest_checkpoint)
        print(f"Found checkpoint: {resume_from_checkpoint}")
    else:
        print("No checkpoints found, starting from scratch")
else:
    print("Output directory doesn't exist, starting from scratch")

# --- 1. Load Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check device availability
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS (Metal Performance Shaders)")
else:
    device = "cpu"
    print("Using CPU")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# --- 2. Configure PEFT (LoRA) ---
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 3. Load and Process Dataset ---
def tokenize_function(examples):
    """Tokenize the formatted text."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=256,
        return_tensors=None
    )

# Load datasets
train_dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
validation_dataset = load_dataset("json", data_files="data/validate.jsonl", split="train")

# Format and tokenize datasets
train_dataset = train_dataset.map(lambda x: {"text": format_prompt(x)})
validation_dataset = validation_dataset.map(lambda x: {"text": format_prompt(x)})

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval = validation_dataset.map(tokenize_function, batched=True, remove_columns=validation_dataset.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# --- 4. Enhanced Training Arguments ---
# Read explanation in README.md
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=100,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,  # Has to be a multiple of eval_steps
    save_total_limit=5,  # Keep more checkpoints
    save_on_each_node=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=0,
    report_to=[],
    dataloader_pin_memory=False,
    remove_unused_columns=True,
    optim="adamw_torch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    fp16=False,
    # Ensure we save everything needed for resuming
    save_safetensors=True,
    # Resume training state (optimizer, scheduler, etc.)
    ignore_data_skip=False,  # Don't skip data when resuming
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# --- 5. Start Training with Resume Support ---
print("Starting training...")
try:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print("Training completed successfully!")
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current state...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    print("You can resume training by running this script again.")
    
except Exception as e:
    print(f"Training failed with error: {e}")
    # Still try to save what we have
    try:
        trainer.save_model()
        print("Partial model saved for debugging.")
    except:
        pass