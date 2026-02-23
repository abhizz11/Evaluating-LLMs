import csv 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset

# GSM8K dataset for quality evaluation
ds = load_dataset("mkurman/gsm8k-SynthLabs-reasoning", split="train[:50]")


# The model and its tokenizer from transformers library
model = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# To store the result
res = []

print(f"\nEvaluating {len(ds)} questions .....\n" + "="*50)

# Feeding the LLM with 50 questions
for i, item in enumerate(ds):
    ans = False
    query = item["query"]
    expected_answer = str(item["ground_truth_extracted"]) # Expected Answer

    # prompt
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Solve the problem step-by-step. End your reasoning with the final number."},
        {"role":"user", "content":query}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt = True
    )

    outputs = pipe(
        prompt,
        max_new_tokens= 512,
        do_sample = False,
        temperature=None,
        top_p=None
    )

    generated_text = outputs[0]["generated_text"][len(prompt):].strip()

    # Answer is set True only if there's a numerical answer in the generated text
    if expected_answer in generated_text:
        ans = True

    res.append([query, generated_text, expected_answer, ans])

# Exporting this to a csv file
csv_file_path = 'output.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(res)

