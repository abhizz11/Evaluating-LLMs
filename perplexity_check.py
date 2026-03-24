import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


# Loading and Tokenizing the model
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
tokenizer.pad_token = tokenizer.eos_token


# Loading the dataset and selecting the first 20 valid texts
print("Loading dataset...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
valid_texts = [text for text in ds["text"] if len(text.strip()) > 50][:20]


# Forked this function from geeksforgeeks
def compute_perplexity_for_batch(input_texts):
    inputs = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True
    )

    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :] 
    shift_labels = input_ids[:, 1:] 

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.dtype)
    negative_log_likelihood = -target_log_probs.sum(dim=-1) / attention_mask[:, 1:].sum(dim=-1)
    perplexities = torch.exp(negative_log_likelihood)
    mean_perplexity_score = torch.mean(perplexities)

    return {
        "perplexities": perplexities.tolist(),
        "mean_perplexity": mean_perplexity_score.item()
    }

perplexities = compute_perplexity_for_batch(valid_texts)
res = []

for key in perplexities:
    if key == "perplexities":
        for i, text in enumerate(valid_texts):
            res.append([text, perplexities[key][i]])

res.append(["Mean Perplexity", perplexities["mean_perplexity"]])

# Printing out the results
print(res)

# Saving the results to a CSV file
csv_file_path = 'perplexity.csv'
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(res)
