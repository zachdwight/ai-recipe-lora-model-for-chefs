# LoRA Recipe Fine-tuning for AI Cooking Companion

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

### Prepare Your Dataset

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


### Configure the Script

Open the `recipe_training.py` script and modify the following variables in the "Configuration" section:

* **`BASE_MODEL_ID`**: Replace `"prithivMLmods/Llama-Express.1-Tiny"` with the actual Hugging Face model ID you wish to fine-tune. Ensure you have access to the model (e.g., you might need to be logged into Hugging Face if it's a gated model).
    * Examples: `"meta-llama/Llama-2-7b-hf"`, `"mistralai/Mistral-7B-v0.1"`, `"google/gemma-2b"`
* **`DATASET_PATH`**: Set this to the path of your JSON recipe file (e.g., `"./secret_recipes.json"`).
* **`OUTPUT_DIR`**: (Optional) Change the directory where the trained LoRA adapters and training logs will be saved.
* **`MAX_SEQ_LENGTH`**: Adjust the maximum sequence length based on the typical length of your recipes and your GPU's VRAM. Longer sequences require more memory.

### Run the Script

Execute the script from your terminal:

```bash
python recipe_training.py
```

---

### Verify Output
Folder containing the lora_recipe_model will be created in your script directory with the final adapters.  Here is the output of a training and test inference using the new weights (LORA):

```bash
Base model and tokenizer loaded.
LoRA model prepared.
trainable params: 11,272,192 || all params: 1,247,086,592 || trainable%: 0.9039
Loading raw recipe data from: secret_recipes.json...
Loaded 326 raw recipes.
Train dataset size: 309
Eval dataset size: 17

Adding EOS to train dataset: 100%|██| 309/309 [00:00<00:00, 15873.70 examples/s]
Tokenizing train dataset: 100%|██████| 309/309 [00:00<00:00, 1020.52 examples/s]
Truncating train dataset: 100%|█████| 309/309 [00:00<00:00, 93482.40 examples/s]
Adding EOS to eval dataset: 100%|██████| 17/17 [00:00<00:00, 6875.25 examples/s]
Tokenizing eval dataset: 100%|██████████| 17/17 [00:00<00:00, 999.57 examples/s]
Truncating eval dataset: 100%|█████████| 17/17 [00:00<00:00, 9884.00 examples/s]

Starting training...
{'loss': 2.3365, 'grad_norm': 1.2229989767074585, 'learning_rate': 3.6e-05, 'num_tokens': 34831.0, 'mean_token_accuracy': 0.4993888534605503, 'epoch': 0.26}
{'loss': 2.0914, 'grad_norm': 1.0498864650726318, 'learning_rate': 7.6e-05, 'num_tokens': 72116.0, 'mean_token_accuracy': 0.5510187316685915, 'epoch': 0.52}
{'loss': 1.7886, 'grad_norm': 0.8332004547119141, 'learning_rate': 0.000116, 'num_tokens': 108671.0, 'mean_token_accuracy': 0.6135538514703512, 'epoch': 0.78}
{'eval_loss': 1.5332924127578735, 'eval_runtime': 2.379, 'eval_samples_per_second': 7.146, 'eval_steps_per_second': 1.261, 'eval_num_tokens': 142032.0, 'eval_mean_token_accuracy': 0.6722870469093323, 'epoch': 1.0}
{'loss': 1.6928, 'grad_norm': 0.7861126661300659, 'learning_rate': 0.00015600000000000002, 'num_tokens': 145690.0, 'mean_token_accuracy': 0.6289788283310928, 'epoch': 1.03}
{'loss': 1.5521, 'grad_norm': 0.9274085760116577, 'learning_rate': 0.000196, 'num_tokens': 182600.0, 'mean_token_accuracy': 0.6557550512254238, 'epoch': 1.28}
{'loss': 1.513, 'grad_norm': 0.9608930349349976, 'learning_rate': 0.00019122695526648968, 'num_tokens': 219688.0, 'mean_token_accuracy': 0.6569658406078815, 'epoch': 1.54}
{'loss': 1.491, 'grad_norm': 1.018894910812378, 'learning_rate': 0.0001628712866590885, 'num_tokens': 255775.0, 'mean_token_accuracy': 0.6601437494158745, 'epoch': 1.8}
{'eval_loss': 1.4084537029266357, 'eval_runtime': 2.3659, 'eval_samples_per_second': 7.185, 'eval_steps_per_second': 1.268, 'eval_num_tokens': 284064.0, 'eval_mean_token_accuracy': 0.6926116347312927, 'epoch': 2.0}
{'loss': 1.4377, 'grad_norm': 0.8362807631492615, 'learning_rate': 0.00012094402627661447, 'num_tokens': 291704.0, 'mean_token_accuracy': 0.6654228575817951, 'epoch': 2.05}
{'loss': 1.3003, 'grad_norm': 0.9704830646514893, 'learning_rate': 7.449572314383237e-05, 'num_tokens': 328003.0, 'mean_token_accuracy': 0.6897008948028087, 'epoch': 2.31}
{'loss': 1.2868, 'grad_norm': 0.9143216609954834, 'learning_rate': 3.355285265611784e-05, 'num_tokens': 364537.0, 'mean_token_accuracy': 0.696876784414053, 'epoch': 2.57}
{'loss': 1.2985, 'grad_norm': 1.0165992975234985, 'learning_rate': 6.953470369291348e-06, 'num_tokens': 402145.0, 'mean_token_accuracy': 0.6931805118918419, 'epoch': 2.83}
{'eval_loss': 1.3955949544906616, 'eval_runtime': 2.3687, 'eval_samples_per_second': 7.177, 'eval_steps_per_second': 1.267, 'eval_num_tokens': 426096.0, 'eval_mean_token_accuracy': 0.7000804543495178, 'epoch': 3.0}
{'train_runtime': 394.2506, 'train_samples_per_second': 2.351, 'train_steps_per_second': 0.297, 'train_loss': 1.598039056500818, 'epoch': 3.0}                  
100%|█████████████████████████████████████████| 117/117 [06:34<00:00,  3.37s/it]

Training complete.
Saving LoRA adapters to ./lora_recipe_model/final_adapters...
LoRA adapters saved.

--- Example Inference ---
Loading base model for inference...
Loading LoRA adapters from ./lora_recipe_model/final_adapters...
Merging LoRA adapters into base model...
LoRA adapters merged.

Generating recipe...

--- Generated Recipe ---
### Instruction: Create a detailed recipe for a quick and easy pasta dish.
### Input: Here is the recipe name and a brief description: "Quick Garlic Pasta" - "A simple, flavorful weeknight meal."
### Response: **Quick Garlic Pasta**

**Description:** A simple, flavorful weeknight meal.

**Ingredients:**
- 400g pasta of your choice
- 2 garlic cloves, peeled
- 1 tbsp olive oil
- 2 red onions, finely chopped
- 1 tbsp red wine vinegar
- 1 tbsp tomato puree
- 2 tbsp sunflower oil
- 1 tsp dried oregano
- 1 tsp dried basil
- 2 tbsp chopped fresh parsley
- 2 bay leaves
- 1 garlic clove, crushed
- 1 tbsp olive oil
- 1 large red onion, finely chopped
- 2 tbsp sunflower oil
- 1 tbsp olive oil
- 1 tbsp white wine vinegar
- 1 tbsp olive oil
- 1 tbsp chopped fresh parsley
- 1 tbsp dried basil
- 4-6 fresh basil leaves

**Instructions:** 
- Cook pasta according to pack instructions.
- Meanwhile, heat the garlic, oil, onions and vinegar in a saucepan.
- Cook for 10-15 mins or until the onions are softened and fragrant.
- Add the tomato puree and simmer for 1 min more.
- Stir in the sunflower oil, oregano, basil, crushed garlic, and bay leaves.
- Season and cook for 2-3 mins.
- Drain the pasta and return to the pan, toss everything together.
- Add the safflower oil, vinegar, parsley and basil, season and cook for 1 min more.
- Serve with the pasta.
--- End Generated Recipe ---

```

Output Conclusion: 

Obviously the model needs some fine tuning and more data as you'll notice the above generated recipe includes multiple tbsps of olive oil!  Note, I also edited the instructions generated output so it would look nice in readme.md as code puts it all on one line.  

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
