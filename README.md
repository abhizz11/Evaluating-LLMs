* Work in Progress will update it properly once everything is done* 

This Python script evaluates the mathematical reasoning capabilities of the Llama-3.2-1B-Instruct model using a subset of the GSM8K dataset.

Key Features
Dataset: Loads the first 50 samples from the mkurman/gsm8k-SynthLabs-reasoning dataset.
Model: Utilizes meta-llama/Llama-3.2-1B-Instruct via the Hugging Face transformers pipeline.
Precision: Runs the model in bfloat16 on a CUDA-enabled GPU for optimized performance.
Evaluation Logic: * Prompts the model as a "helpful math assistant."
Checks if the expected ground-truth answer is present within the model's generated response.

Output: Saves the results (query, generated text, expected answer, and pass/fail status) to a file named output.csv.

Dependencies
To run this code, you will need:
torch
transformers
datasets
accelerate (for GPU loading)
