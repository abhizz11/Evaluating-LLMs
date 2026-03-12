import csv
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset

# Loading the datset
ds = load_dataset("knkarthick/samsum", split="train[:50]")

# Loading and Tokenizing the model
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Loading the ROUGE metric
rouge = evaluate.load("rouge")

res = [] # To store results for CSV output
conversations = [] # Conversations for Llama input
llama_summaries = [] # Llama's Summaries
human_summary = [] # Human Summaries

print("Loading dataset...")

for i, item in enumerate(ds):
    source_text = item["dialogue"]
    reference_summary = item["summary"]

    # Prompt engineering for Llama
    messages = [
        {"role": "system", "content": "You are a professional summarizer. "
        "Summarize the given dialogue in a concise manner. "
        "Output ONLY the summary. "
        "Do not add any explanations or conversational filler."},
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
        do_sample = False,
        return_full_text=False
    )

    # Extracting only the generated summary without the prompt
    generated_summary = outputs[0]["generated_text"].strip()
    conversations.append(source_text)
    llama_summaries.append(generated_summary)
    human_summary.append([reference_summary])

    # ROUGE EVALUATION
    rouge_scores = rouge.compute(predictions=[generated_summary], references=[[reference_summary]])  


    res.append([source_text, generated_summary, reference_summary, rouge_scores['rouge1'], rouge_scores['rouge2'], rouge_scores['rougeL']])

csv_file_path = "rouge_results.csv"
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Source Text (EN)", "Generated Summary", "Reference Summary", "ROUGE-1", "ROUGE-2", "ROUGE-L"])
    writer.writerows(res)

print(f"Results saved to {csv_file_path}")