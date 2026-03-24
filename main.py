import csv 
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset



# Function to load the model and tokenizer
def loading_model(model_name):
	print(f"Loading model {model_name}...")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
	pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

	return [tokenizer, pipe]


# GSM8K dataset for quality evaluation
def evaluate_gsm8k(tokenizer, pipe):
    res = []
    ds = load_dataset("mkurman/gsm8k-SynthLabs-reasoning", split="train[:20]")
    
    # Define a generator to prepare prompts for the pipeline
    def data_generator():
        for item in ds:
            messages = [
                {"role": "system", "content": "You are a helpful math assistant. Solve the problem step-by-step. End your reasoning with the final number."},
                {"role": "user", "content": item["query"]}
            ]
            yield tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process in batches
    outputs = pipe(
        data_generator(), 
        batch_size=4, 
        max_new_tokens=512, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    for i, out in enumerate(outputs):
        prompt = ds[i]["query"]
        generated_text = out[0]["generated_text"]
        
        # Logic to extract only the new response
        response_only = generated_text.split("assistant\n")[-1].strip()
        
        expected_answer = str(ds[i]["ground_truth_extracted"])
        ans = expected_answer in response_only
        res.append([prompt, response_only, expected_answer, ans])
    
    return res
        

# Translation evaluation using BLEU with batched inference
def evaluate_translation_bleu(tokenizer, pipe):
    # Load metric and dataset
    bleu = evaluate.load("bleu")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train[:50]")
    results = []
    predictions = []
    references = []
    sources = []

    # 1. Prepare prompts using a generator for memory efficiency
    def prompt_generator():
        for item in ds:
            en_text = item["translation"]["en"]
            sources.append(en_text)
            references.append(item["translation"]["fr"])
            
            messages = [
                {"role": "system", "content": "You are a professional translator. Translate English to French. Output ONLY the translation."},
                {"role": "user", "content": en_text}
            ]
            yield tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 2. Batched Inference (GPU Optimization)
    outputs = pipe(
        prompt_generator(),
        batch_size=8,
        max_new_tokens=256,
        do_sample=False,
        return_full_text=False 
    )

    # 3. Collect predictions
    for out in outputs:
        predictions.append(out[0]["generated_text"].strip())

    for src, pred, ref in zip(sources, predictions, references):
        # references must be a list of lists for BLEU
        ref_list = [ref] 
        
        # Calculate individual n-gram scores
        s2 = bleu.compute(predictions=[pred], references=[ref_list], max_order=2)['bleu']
        s3 = bleu.compute(predictions=[pred], references=[ref_list], max_order=3)['bleu']
        s4 = bleu.compute(predictions=[pred], references=[ref_list], max_order=4)['bleu']
        
        results.append([src, pred, ref, s2, s3, s4])

    return results


# Summarization evaluation using ROUGE with batched inference
def evaluate_summarization_rouge(tokenizer, pipe):
    """
    Evaluates summarization quality using ROUGE scores with batched inference.
    """
    rouge = evaluate.load("rouge")
    ds = load_dataset("knkarthick/samsum", split="train[:50]")
    
    results = []
    predictions = []
    references = []
    dialogues = []

    # 1. Prepare prompts using a generator
    def prompt_generator():
        for item in ds:
            # SAMSum uses 'dialogue' and 'summary'
            text = item["dialogue"]
            dialogues.append(text)
            references.append(item["summary"])
            
            messages = [
                {"role": "system", "content": "You are a professional summarizer. Summarize the dialogue concisely. Output ONLY the summary."},
                {"role": "user", "content": text}
            ]
            yield tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 2. Batched Inference
    # return_full_text=False is critical to avoid manually slicing the prompt
    outputs = pipe(
        prompt_generator(),
        batch_size=8,
        max_new_tokens=256,
        do_sample=False,
        return_full_text=False
    )

    # 3. Collect predictions
    for out in outputs:
        predictions.append(out[0]["generated_text"].strip())

    # 4. Evaluation Phase
    # Compute sentence-level scores for the CSV
    for dial, pred, ref in zip(dialogues, predictions, references):
        # ROUGE usually expects a single string for prediction and reference 
        score = rouge.compute(predictions=[pred], references=[ref])
        
        results.append([
            dial, 
            pred, 
            ref, 
            score['rouge1'], 
            score['rouge2'], 
            score['rougeL'], 
            score['rougeLsum']
        ])

    # 5. Global Corpus ROUGE
    corpus_rouge = rouge.compute(predictions=predictions, references=references)
    print(f"Total Corpus ROUGE-L: {corpus_rouge['rougeL']:.4f}")

    return results


# Perplexity evaluation using a batch processing approach
def compute_perplexity_for_batch(input_texts):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
    tokenizer.pad_token = tokenizer.eos_token
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

# Perplexity Score calculation function
def perplexity():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    valid_texts = [text for text in ds["text"] if len(text.strip()) > 50][:20]
    perplexities = compute_perplexity_for_batch(valid_texts)
    res = []

    for key in perplexities:
        if key == "perplexities":
            for i, text in enumerate(valid_texts):
                res.append([text, perplexities[key][i]])

    res.append(["Mean Perplexity", perplexities["mean_perplexity"]])
    return res

# Write data to CSV file
def write_to_csv(data, csv_file_path):
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

model = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer, pipe = loading_model(model)

# math_results = evaluate_gsm8k(tokenizer, pipe)
# write_to_csv(math_results, "gsm8k_results.csv")
# bleu_results = evaluate_translation_bleu(tokenizer, pipe)
# write_to_csv(bleu_results, "bleu_results.csv")
# rouge_results = evaluate_summarization_rouge(tokenizer, pipe)
# write_to_csv(rouge_results, "rouge_results.csv")

perplexity_results = perplexity()
write_to_csv(perplexity_results, "perplexity_results.csv")




