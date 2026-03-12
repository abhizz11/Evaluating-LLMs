import json
import time
import csv
import torch
from google import genai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset


# Loading the dataset
ds = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train[:20]")


# Loading and Tokenizing the model
model = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

res = [] # To store results for CSV output
predictions = [] # To store Llama's translations
references = [] # To store human translations

client = genai.Client()

# GEMINI EVALUATION FUNCTION
def evaluate_translation(source, reference, prediction):
    prompt = f"""
        Evaluate the following English to French translation.
        Source (EN): {source}
        Reference (FR): {reference}
        Prediction (FR): {prediction}

        Provide a quality score from 1 to 10 and a brief reason. 
        Return ONLY a JSON object with keys "score" and "reason".
        """
    response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.0
            }
        )
    return json.loads(response.text)


for i, item in enumerate(ds):
    source_text = item["translation"]["en"]
    expected_translation = item["translation"]["fr"]

    # Prompt engineering for Llama
    messages = [
        {"role": "system", "content": "You are a professional translator. "
        "Translate the given English text to French. Output ONLY the translated"
        " French text. Do not add any explanations or conversational filler."},
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

    try:
        time.sleep(6)
        gemini_eval = evaluate_translation(source_text, expected_translation, generated_translation)
        gemini_score = gemini_eval.get("score", 0)
        reasoning = gemini_eval.get("reason", "No reason provided.")
    except Exception as e:
        print(f"Gemini error at index{i}: {e}")
        gemini_score, reasoning = 0, "Evaluation failed."
    
    res.append([source_text, generated_translation, expected_translation, gemini_score, reasoning])

csv_file_path = "gemini_eval.csv"
header = ["Source (EN)", "Llama (FR)", "Reference (FR)", "Gemini Score", "Gemini Reasoning"]

with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(res)

print(f"Evaluation complete! Saved to {csv_file_path}")