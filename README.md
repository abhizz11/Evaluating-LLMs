# 📊 Evaluating LLMs: Benchmarking & Judge-Led Evaluation

This repository provides a comprehensive framework for evaluating Large Language Models (LLMs) across multiple NLP tasks including mathematics, translation, summarization, and general knowledge. It combines automated benchmarking with an **LLM-as-a-Judge** system for qualitative scoring.

---

## 🚀 Key Features

* **Multi-Task Evaluation**
  Supports:

  * GSM8K (Mathematics)
  * Samsum (Summarization)
  * Opus-100 (Translation)
  * WikiText (Perplexity)

* **LLM-as-a-Judge**
  Uses *Llama-3.2-1B-Instruct* to automatically grade model outputs based on:

  * Custom rubrics
  * Likert scale scoring (1–5)

* **Performance Optimization**

  * Batched inference
  * bfloat16 precision for efficient GPU utilization

* **Evaluation Metrics**

  * BLEU (n = 2, 3, 4)
  * ROUGE (1, 2, L)
  * Perplexity
  * Accuracy

---

## 📁 File Structure
```
| File             | Description                                                              |
| ---------------- | ------------------------------------------------------------------------ |
| `main.py`        | Centralized script with batched evaluation loops and the universal Judge |
| `bleu_check.py`  | Translation evaluation using BLEU                                        |
| `rouge_check.py` | Summarization evaluation using ROUGE                                     |
| `my_file.py`     | Mathematical reasoning evaluation on GSM8K                               |
```
---

## 🛠️ Getting Started

### Prerequisites

* CUDA-enabled GPU (recommended for performance)

### Installation

```bash
pip install torch transformers datasets evaluate accelerate rouge_score
```

---

## ▶️ Usage

Run the main evaluation pipeline:

```bash
python main.py
```

Results will automatically be exported as `.csv` files:

* `gk_judge_results.csv`
* `bleu_results.csv`
* (and others depending on tasks)

---

## 📊 Evaluation Methodology

This project follows a **dual-layered evaluation approach**:

### 1. Quantitative Evaluation

Standard automated metrics:

* BLEU → Translation quality
* ROUGE → Summarization quality
* Perplexity → Language modeling performance
* Accuracy → Task correctness

### 2. Qualitative Evaluation (LLM-as-a-Judge)

A judge model evaluates:

* Reasoning quality
* Correctness
* Coherence

This enables **human-like scoring**, capturing nuances that traditional metrics may miss.

---

## 📌 Notes

* Designed for extensibility — you can easily plug in new datasets or evaluation metrics
* Optimized for batch processing and scalable experimentation
* Outputs are structured for easy analysis and visualization

---

