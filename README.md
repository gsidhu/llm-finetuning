# LLM Finetuning
This repo contains code for fine tuning an LLM using the LoRA method with PyTorch. It trains an LLM to infer user intent and respond in a structured JSON output.

The training data, system prompt, and almost all of the code has been generated using Claude, Gemini and ChatGPT. So there's that. I am basically a human medium for AI reproduction.

The values are optimised for an up to 1.5B parameter model running an M1 Pro MacBook Pro with 16GB RAM.

The `gguf-py` directory comes from the [llama.cpp repo](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py).

The `convert_hf_to_gguf.py` script also comes from the [llama.cpp repo](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py).

## I. Setup and Configuration
1. Create a virtual environment and install the required packages.
```python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This installs all the packages required for fine tuning (an adapter), merging (with the base model) and converting (to gguf) the model.

2. Prepare a training, testing and validation dataset in the `data` directory. Use the JSONL format.
3. Use `python3 ./pytorch-lora/debug_jsonl.py` to debug formatting errors in training data.
4. Set the environment variables in `.env`. Also set the system prompt while you are at it.

Note: Make sure that the prompt format functions in the [`utils.py`](pytorch-lora/utils.py) script are in line with the format of the [training data](data/train.jsonl) and the [system prompt](./system_prompt.txt).

## II. Fine Tuning
Pick a model you would like to fine tune. Pick a smaller model to begin with. A model name ending with `-Instruct` is well suited for fine tuning. You can find them [on Hugging Face](https://huggingface.co/models?sort=downloads&search=instruct).

Run the fine tuning –
```python
# Download the model set in environment
python3 ./pytorch-lora/download_model.py
# Run the fine tuning (see param explanation below)
python3 ./pytorch-lora/finetune.py
# Merge the fine tuned adapter with the base model
python3 ./pytorch-lora/merge_model.py
# (Optional) Run model evaluation on the test items
python3 ./pytorch-lora/evaluate_model.py
# (Optional) Test the model on a specific prompt
python3 ./pytorch-lora/inference.py
```

<details>
<summary><strong>Training Parameters Explained</strong></summary>
<blockquote>
<strong>Core Training Parameters</strong>

`num_train_epochs=3`
- Number of complete passes through your entire training dataset
- With 3 epochs, the model sees each training example exactly 3 times
- More epochs = more training, but risk of overfitting

`per_device_train_batch_size=2`
- Number of samples processed at once during training
- Set low (2) due to M1 Mac memory constraints
- Smaller batches = less memory usage but noisier gradients

`per_device_eval_batch_size=2`
- Number of samples processed during evaluation
- Usually can be higher than training batch size, but kept same for consistency
- Only affects evaluation speed, not training quality

`gradient_accumulation_steps=8`
- **Key parameter!** Accumulates gradients over 8 steps before updating weights
- Effective batch size = per_device_train_batch_size × gradient_accumulation_steps = 2 × 8 = 16
- Simulates larger batch training without using more memory
- Higher values = more stable gradients but slower updates

**Learning Parameters**

`warmup_steps=100`
- Gradually increases learning rate from 0 to learning_rate over first 100 steps
- Prevents early training instability from large learning rates
- Critical for large language models

`learning_rate=2e-4`
- Maximum learning rate (0.0002)
- Higher for LoRA fine-tuning than full fine-tuning
- Controls how big steps the optimizer takes

**Monitoring & Evaluation**

`logging_steps=10`
- Prints training metrics (loss, learning rate) every 10 steps
- More frequent = better monitoring but more console output

`eval_strategy="steps"`
- Run evaluation based on steps (not epochs)
- Alternative: "epoch" would evaluate after each complete epoch

`eval_steps=100`
- Run evaluation every 100 training steps
- Helps monitor overfitting and model performance during training

**Checkpointing Parameters**

`save_steps=50`
- Save model checkpoint every 50 steps
- More frequent = better recovery options but more disk usage

`save_total_limit=5`
- Keep only the 5 most recent checkpoints
- Older checkpoints are automatically deleted
- Balances safety with disk space

`save_on_each_node=True`
- Saves checkpoints on each compute node (not relevant for single-machine training)
- Good practice to keep enabled

**Model Selection**

`load_best_model_at_end=True`
- After training completes, loads the checkpoint with the best evaluation metric
- Without this, you get the model from the final step (which might not be the best)
- Note: Setting load_best_model_at_end to True requires the `save_steps` to be a round multiple of the `eval_steps`

`metric_for_best_model="eval_loss"`
- Uses evaluation loss to determine "best" model
- Lower loss = better model

`greater_is_better=False`
- Since we're using loss (lower is better), set to False
- Would be True for metrics like accuracy

`dataloader_num_workers=0`
- Number of CPU processes for data loading
- Set to 0 on M1 Mac to avoid multiprocessing issues
- Higher values can speed up data loading on other systems

**Logging & Reporting**

`report_to=[]`
- Controls which experiment tracking services to log metrics to
- Empty list implies no external logging
- Possible values: ["wandb", "tensorboard", "comet_ml", "mlflow", "neptune"]

**Memory Optimization**

`dataloader_pin_memory=False`
- When True, pins data tensors in CPU memory for faster GPU transfer
- Disabled for M1 Mac because:
  - MPS (Metal Performance Shaders) doesn't benefit from pinned memory like CUDA
  - Can actually cause memory issues on unified memory systems
  - On CUDA systems, this would typically be True for faster CPU → GPU transfers

`remove_unused_columns=True`
- Automatically removes dataset columns not used by the model
- Saves memory and prevents potential conflicts
- Your dataset might have extra columns like metadata that the model doesn't need
- Good practice to keep enabled unless you need those columns for custom logic

**Optimizer Configuration**

`optim="adamw_torch"`
- Specifies which optimizer implementation to use
- AdamW is the gold standard for transformer fine-tuning
- PyTorch version is well-optimized and stable

**Memory-Saving Techniques**

`gradient_checkpointing=True`
- Critical for memory savings! Trades computation for memory
- Instead of storing all intermediate activations, recomputes them during backward pass
- Can reduce memory usage by 30-50% with ~10-20% training slowdown
- Essential for fitting 1.5B models on 16GB RAM. Without this, you'd likely run out of memory

`gradient_checkpointing_kwargs={'use_reentrant': False}`
- Controls how gradient checkpointing is implemented
- use_reentrant=False uses newer, more stable implementation
- The older reentrant version can cause issues with some model architectures
- This setting prevents the "grad can be implicitly created only for scalar outputs" error
- Always use False with modern PyTorch versions

**Precision Settings**

`fp16=False`
- Disables 16-bit floating point training
- Disabled because M1 Mac MPS doesn't fully support FP16 training
- On NVIDIA GPUs, FP16 would: Halve memory usage; Speed up training significantly; Require careful loss scaling to prevent underflow

</blockquote>
</details> 

## III. Running in Ollama
1. Convert the PyTorch model to GGUF for use with Ollama/llama.cpp –
```
python gguf-py/convert_hf_to_gguf.py ./intent-classifier-final/ \              
  --outfile ./intent-classifier-final.f16.gguf \
  --outtype f16
```

2. (Optional) Quantize –
```
llama-quantize ./intent-classifier-final.f16.gguf ./intent-classifier-final.Q4_K_M.gguf Q4_K_M
```

Make sure you have [llama.cpp installed](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md). I prefer homebrew.

3. Create the Ollama modelfile ([see example](/qwen-voice-assistant-modefile))

4. Load the modelfile in Ollama –
```
ollama create buzee-voice-assistant -f ./qwen-voice-assistant-modefile
```

Check that the model is loaded using `ollama list`.

5. Run the model in Ollama –
```
ollama run buzee-voice-assistant
```