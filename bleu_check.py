import csv
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset

# Loading the dataset
ds = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train[:50]")

# Loading and Tokenizing the model
model = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Loading the BLEU metric
bleu = evaluate.load("bleu")

res = [] # To store results for CSV output
predictions = [] # To store Llama's translations
references = [] # To store human translations

print("Loading dataset...")

for i, item in enumerate(ds):
    source_text = item["translation"]["en"]
    expected_translation = item["translation"]["fr"]

    # Prompt engineering for Llama
    messages = [
        {"role": "system", "content": "You are a professional translator. Translate the given English text to French. Output ONLY the translated French text. Do not add any explanations or conversational filler."},
        {"role": "user", "content": source_text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )

    outputs = pipe(
        prompt, 
        max_new_tokens = 256,
        do_sample = False
    )

    generated_translation = outputs[0]["generated_text"][len(prompt):].strip()
    predictions.append(generated_translation)
    references.append([expected_translation])

    # BLEU EVALUATION
    n2 = bleu.compute(predictions=[generated_translation], references=[[expected_translation]], max_order=2)  
    n3 = bleu.compute(predictions=[generated_translation], references=[[expected_translation]], max_order=3)
    n4 = bleu.compute(predictions=[generated_translation], references=[[expected_translation]], max_order=4)


    res.append([source_text, generated_translation, expected_translation, n2['bleu'], n3['bleu'], n4['bleu']])


csv_file_path = "bleu_results.csv"
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Source Text (EN)", "Generated Translation (FR)", "Reference Translation (FR)", "BLEU Score (n=2)", "BLEU Score (n=3)", "BLEU Score (n=4)"])
    writer.writerows(res)

print(f"Results saved to {csv_file_path}")