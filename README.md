# LoRA Recipe Fine-tuning

This script demonstrates how to fine-tune a Causal Language Model (CLM) using Low-Rank Adaptation (LoRA) with the `transformers`, `peft`, and `trl` libraries. The goal is to adapt a pre-trained model to generate detailed recipes based on provided instructions and input context.

---

## Features

* **4-bit Quantization:** Loads the base model in 4-bit to significantly reduce VRAM usage.
* **LoRA Fine-tuning:** Efficiently fine-tunes a large language model using the LoRA technique.
* **Instruction-Tuning Format:** Preprocesses recipe data into a standard instruction-tuning format suitable for language model training.
* **`SFTTrainer`:** Utilizes the `trl.SFTTrainer` for simplified supervised fine-tuning.
* **Gradient Accumulation & Checkpointing:** Employs techniques to train with larger effective batch sizes even on memory-constrained GPUs.
* **Automatic Device Mapping:** Leverages `device_map="auto"` for efficient resource utilization across available GPUs.
* **Inference Example:** Includes a section to demonstrate how to load the fine-tuned LoRA adapters and generate new recipes.

---

## Setup

### Prerequisites

* Python 3.11 or higher
* NVIDIA GPU with CUDA support for accelerated training (recommended)

### Installation

1.  **Create a Virtual Environment (Recommended):**

    ```bash
    python3.11 -m venv ~/venvs/recipe_env
    source ~/venvs/recipe_env/bin/activate
    ```

2.  **Install Dependencies:**

    ```bash
    pip install torch
    pip install transformers==4.53.1
    pip install bitsandbytes
    pip install datasets
    pip install peft
    pip install trl
    ```

    ---

## Usage

### 1. Prepare Your Dataset

Create a JSON file containing an array of recipe objects. Each recipe object **must** contain the following keys: `"Name"`, `"Description"`, `"Ingredients"` (as a list of strings), and `"Method"` (as a list of strings).

**Example `secret_recipes.json`:**

```json
[
  {
    "Name": "Classic Spaghetti Carbonara",
    "Description": "A timeless Italian pasta dish with eggs, hard cheese, cured pork, and black pepper.",
    "Ingredients": [
      "200g spaghetti",
      "100g guanciale (or pancetta/bacon)",
      "2 large eggs",
      "50g Pecorino Romano cheese, grated",
      "Freshly ground black pepper",
      "Salt to taste"
    ],
    "Method": [
      "Bring a large pot of salted water to a boil. Add the spaghetti and cook according to package directions until al dente.",
      "While the pasta cooks, cut the guanciale into small cubes. In a large skillet, cook the guanciale over medium heat until crispy. Remove the guanciale and set aside, leaving the rendered fat in the pan.",
      "In a bowl, whisk together the eggs, grated Pecorino Romano, and a generous amount of black pepper.",
      "Quickly pour the egg mixture over the hot pasta, stirring vigorously to create a creamy sauce. Add a splash of reserved pasta water as needed to achieve the desired consistency.",
      "Stir in the crispy guanciale. Serve immediately, garnished with extra Pecorino Romano and black pepper."
    ]
  },
  {
    "Name": "Simple Green Smoothie",
    "Description": "A healthy and refreshing smoothie packed with greens and fruit.",
    "Ingredients": [
      "1 cup spinach",
      "1/2 banana, frozen",
      "1/2 cup unsweetened almond milk",
      "1/4 cup pineapple chunks, frozen",
      "1 tablespoon chia seeds"
    ],
    "Method": [
      "Combine all ingredients in a blender.",
      "Blend until smooth and creamy.",
      "Pour into a glass and enjoy immediately."
    ]
  }
]
```
---

**Chunk 4: Usage (Part 2: Script Configuration and Run)**


### 2. Configure the Script

Open the `finetune_recipe.py` script and modify the following variables in the "Configuration" section:

* **`BASE_MODEL_ID`**: Replace `"prithivMLmods/Llama-Express.1-Tiny"` with the actual Hugging Face model ID you wish to fine-tune. Ensure you have access to the model (e.g., you might need to be logged into Hugging Face if it's a gated model).
    * Examples: `"meta-llama/Llama-2-7b-hf"`, `"mistralai/Mistral-7B-v0.1"`, `"google/gemma-2b"`
* **`DATASET_PATH`**: Set this to the path of your JSON recipe file (e.g., `"./secret_recipes.json"`).
* **`OUTPUT_DIR`**: (Optional) Change the directory where the trained LoRA adapters and training logs will be saved.
* **`MAX_SEQ_LENGTH`**: Adjust the maximum sequence length based on the typical length of your recipes and your GPU's VRAM. Longer sequences require more memory.

### 3. Run the Script

Execute the script from your terminal:

```bash
python your_script_name.py
```

---

## Troubleshooting

* **Out of Memory (OOM) Errors:**
    * Reduce `per_device_train_batch_size`.
    * Increase `gradient_accumulation_steps`.
    * Decrease `MAX_SEQ_LENGTH`.
    * If using `torch.float16`, try `torch.bfloat16` if your GPU supports it (newer NVIDIA cards like RTX 30/40 series, A100).
* **`bitsandbytes` Errors:** Ensure you have a compatible CUDA version and `bitsandbytes` installation. Sometimes reinstalling with `pip install bitsandbytes` can resolve issues.
* **Model Access Issues:** If you're using a gated model from Hugging Face (e.g., Llama-2), make sure you have accepted its terms of use and are logged in to Hugging Face CLI: `huggingface-cli login`.
* **No recipes generated/Poor quality recipes:**
    * Increase `num_train_epochs`.
    * Adjust `learning_rate`.
    * Ensure your `DATASET_PATH` is correct and the JSON structure matches expectations.
    * Experiment with `lora_config` parameters (e.g., `r`, `lora_alpha`, `target_modules`).
