
# python3.11 -m venv ~/venvs/recipe_env
# source ~/venvs/recipe_env/bin/activate

# pip install torch
# pip install transformers==4.53.1
# pip install bitsandbytes
# pip install datasets
# pip install peft
# pip install trl

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
import json
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset # Import Dataset directly for creating from list
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


print("transformers version:", transformers.__version__)
from transformers import TrainingArguments
print("TrainingArguments init args:", TrainingArguments.__init__.__code__.co_varnames)

# --- Configuration ---
# IMPORTANT: Replace with your actual model name from Hugging Face Hub
# Examples: "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "google/gemma-2b"
# Ensure you have access to the model (e.g., logged into Hugging Face if gated)
BASE_MODEL_ID = "prithivMLmods/Llama-Express.1-Tiny" # Example for demonstration

# Replace with the path to your .json file containing an array of recipe objects
DATASET_PATH = "secret_recipes.json" # Assumes a single .json file

# This is the column in your final Hugging Face Dataset that contains the full prompt-response string
TEXT_FIELD = "text"

OUTPUT_DIR = "./lora_recipe_model"
MAX_SEQ_LENGTH = 512  # Max length of tokenized input sequence. Adjust based on your recipe length.
                      # Longer sequences require more VRAM.

# --- Step 2: Load Model and Tokenizer ---
print(f"Loading base model: {BASE_MODEL_ID}...")
# Load the model in 4-bit for memory efficiency.
# Using torch.bfloat16 if your GPU supports it (e.g., A100, RTX 30 series and newer)
# otherwise use torch.float16.
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16, # or torch.float16 if bfloat16 is not supported by your GPU
    device_map="auto",  # Automatically place layers on devices
    offload_folder="./offload",  # Folder where CPU offloaded tensors are stored
    offload_state_dict=True 
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# Important: Set pad_token for generation if the model doesn't have one (e.g., Llama-2)
# Mistral and many others use EOS as pad_token by default.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id # Ensure ID is also set

print("Base model and tokenizer loaded.")

# --- Step 3: Configure LoRA ---

# Prepare the model for k-bit training (4-bit or 8-bit)
# This enables gradient checkpointing and prepares the model for PEFT.
model.config.use_cache = False # Disable cache to use gradient checkpointing
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,  # LoRA attention dimension (rank) - common values 8, 16, 32, 64
    lora_alpha=32,  # Alpha parameter for LoRA scaling - usually 2*r
    target_modules=[ # These are typical target modules for Llama/Mistral/Gemma models
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # "lm_head", # Only include if you want to fine-tune the final output layer (less common with LoRA)
    ],
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none",  # Don't train bias terms
    task_type="CAUSAL_LM", # Specify the task type
)

# Apply LoRA configuration to the model
model = get_peft_model(model, lora_config)

print("LoRA model prepared.")
print(model.print_trainable_parameters()) # See how many parameters are trainable

# --- Step 1 (Cont.): Load and Preprocess Data ---
print(f"Loading raw recipe data from: {DATASET_PATH}...")

# Load the entire JSON file into a Python list of dictionaries
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    raw_recipe_data = json.load(f)

print(f"Loaded {len(raw_recipe_data)} raw recipes.")

formatted_recipes = []

for recipe in raw_recipe_data:
    # Ensure all required fields exist to prevent KeyError
    name = recipe.get("Name", "Untitled Recipe")
    description = recipe.get("Description", "No description provided.")
    ingredients = recipe.get("Ingredients", [])
    method = recipe.get("Method", [])

    # Construct the ingredients string
    ingredients_str = "\n".join([f"- {ing}" for ing in ingredients])

    # Construct the method string with numbered steps
    method_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(method)])

    # Construct the full response part
    full_response = f"**{name}**\n\n" \
                    f"**Description:** {description}\n\n" \
                    f"**Ingredients:**\n{ingredients_str}\n\n" \
                    f"**Instructions:**\n{method_str}"

    # Construct the instruction and input
    instruction = f"Create a detailed recipe for {name}."
    input_context = f"Here is the recipe name and a brief description: \"{name}\" - \"{description}\""

    # Combine into the final instruction-tuning format for the 'text' field
    formatted_entry = {
        TEXT_FIELD: f"### Instruction: {instruction}\n"
                    f"### Input: {input_context}\n"
                    f"### Response: {full_response}"
    }
    formatted_recipes.append(formatted_entry)

# Convert your list of dictionaries into a Hugging Face Dataset object
dataset = Dataset.from_list(formatted_recipes)

# Split the dataset into train and validation
train_val_split = dataset.train_test_split(test_size=0.05, seed=42) # 5% for validation
train_dataset = train_val_split["train"]
eval_dataset = train_val_split["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")
print("First example (formatted for training) in training dataset:")
print(train_dataset[0][TEXT_FIELD])
torch.cuda.empty_cache()
# --- Step 4: Fine-tuning ---

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Adjust based on your GPU memory. Lower if OOM.
    gradient_accumulation_steps=8,  # Accumulate gradients over 4 steps to simulate a batch size of 8 (2*4)
    warmup_steps=50,                # Linear warmup for the first 50 steps
    num_train_epochs=3,             # Number of full passes through the training data
    learning_rate=2e-4,             # Initial learning rate for AdamW optimizer
    fp16=False,bf16=True,               # Enable mixed precision training (recommended if supported by GPU)
    logging_steps=10,               # Log training metrics every 10 steps
    output_dir=OUTPUT_DIR,          # Directory to save checkpoints and logs
    optim="paged_adamw_8bit",       # Optimizer optimized for 8-bit quantization (good for memory)
    lr_scheduler_type="cosine",     # Cosine learning rate scheduler
    save_strategy="epoch",          # Save checkpoint after each epoch
    eval_strategy="epoch",    # Evaluate after each epoch
    load_best_model_at_end=True,    # Load the best model based on evaluation loss at the end of training
    report_to="tensorboard",        # Report metrics to TensorBoard for visualization
    remove_unused_columns=False,
    max_grad_norm=0.3,
    gradient_checkpointing=True
)

# Initialize SFTTrainer (Supervised Fine-Tuning Trainer)
# SFTTrainer simplifies instruction tuning and handles packing sequences.
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,        # Pass the LoraConfig
    #dataset_text_field=TEXT_FIELD,  # The column containing the text to train on
    #max_seq_length=MAX_SEQ_LENGTH,  # Max sequence length for tokenization and packing
    #tokenizer=tokenizer,
    args=training_args,
    #packing=True,                   # Packs multiple short examples into a single long sequence to improve GPU utilization
)

# Start training
print("\nStarting training...")
trainer.train()
print("Training complete.")

# --- Step 5: Save the LoRA adapters ---
# The trained LoRA adapter weights are saved, not the full model.
print(f"Saving LoRA adapters to {OUTPUT_DIR}/final_adapters...")
trainer.save_model(f"{OUTPUT_DIR}/final_adapters")
print("LoRA adapters saved.")

# --- Example for Inference (how to load and use the fine-tuned model) ---
print("\n--- Example Inference ---")
try:
    from peft import PeftModel
    print("Loading base model for inference...")
    inference_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    print(f"Loading LoRA adapters from {OUTPUT_DIR}/final_adapters...")
    inference_model = PeftModel.from_pretrained(inference_model, f"{OUTPUT_DIR}/final_adapters")

    # Optional: Merge LoRA weights into the base model for easier deployment
    # This creates a single, consolidated model (requires more VRAM temporarily)
    print("Merging LoRA adapters into base model...")
    inference_model = inference_model.merge_and_unload()
    print("LoRA adapters merged.")

    inference_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if inference_tokenizer.pad_token is None:
        inference_tokenizer.pad_token = inference_tokenizer.eos_token

    inference_model.eval() # Set model to evaluation mode

    prompt_text = "### Instruction: Create a detailed recipe for a quick and easy pasta dish.\n### Input: Here is the recipe name and a brief description: \"Quick Garlic Pasta\" - \"A simple, flavorful weeknight meal.\"\n### Response:"
    input_ids = inference_tokenizer(prompt_text, return_tensors="pt").input_ids.to(inference_model.device)

    print("\nGenerating recipe...")
    outputs = inference_model.generate(
        input_ids,
        max_new_tokens=512, # Max tokens to generate
        num_beams=1,        # Use greedy decoding for simplicity
        do_sample=True,     # Sample for more creative output
        temperature=0.7,    # Controls randomness
        top_p=0.9,          # Controls diversity
        eos_token_id=inference_tokenizer.eos_token_id,
        pad_token_id=inference_tokenizer.pad_token_id
    )
    generated_text = inference_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Generated Recipe ---")
    print(generated_text)
    print("--- End Generated Recipe ---")

except Exception as e:
    print(f"\nCould not run inference example. This is common if you have limited VRAM. Error: {e}")
    print("To perform inference, ensure your GPU has enough memory to load both the base model and adapters, or use `merge_and_unload()` on a machine with more resources.")
