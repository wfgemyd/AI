# GPT-2 Zero-Shot Evaluation on NLP Benchmarks

This repository contains Python code to evaluate GPT‑2 on several zero-shot NLP benchmarks, following the methodology described in the original paper: Language Models are Unsupervised Multitask Learners (Radford et al., 2019). The goal is to reproduce or compare (in a meaningful way) the performance of GPT‑2 against the results reported in the paper. Using the Hugging Face Transformers library and the Datasets library, we evaluate GPT‑2 on the following tasks:

LAMBADA: Tests the model’s ability to predict the last word in a passage.
WikiText2: Measures perplexity (lower is better).
PIQA: Tests physical commonsense reasoning in a multiple-choice setting.
HellaSwag: Tests commonsense inference in a multiple-choice setting.

Requirements:
Python: 3.7+
PyTorch: (with CUDA if you have a GPU)
Transformers
Datasets

You can install everything via:
pip install torch transformers datasets
Note: For GPU usage, ensure you have the CUDA-enabled version of PyTorch and the required NVIDIA drivers.

Files and Scripts:
LAMBADA.py:

Evaluates GPT‑2 accuracy on the LAMBADA dataset.
Checks if the model correctly predicts the very last token in each passage.
Prints the final accuracy to the console and writes the result to a .txt file.
Example usage:
python LAMBADA.py

WikiText2.py:

Computes perplexity on the WikiText2 (test split).
Prints the final perplexity and writes it to a .txt file.
Example usage:
python WikiText2.py

PIQA.py:

Evaluates GPT‑2 in a multiple-choice, zero-shot setting on the PIQA dataset.
Each example has two possible solutions; the script compares the log-likelihood of each solution and picks the higher.
Prints the final accuracy and writes it to a .txt file.
Important: Since PIQA contains custom loading code, you must pass trust_remote_code=True in the load_dataset function; the script includes this fix.
Example usage:
python PIQA.py

HellaSwag.py:

Evaluates GPT‑2 on HellaSwag (multiple-choice, zero-shot).
Each example has four possible endings; the script calculates log-likelihoods and picks the highest.
Prints the final accuracy and writes it to a .txt file.
Also requires trust_remote_code=True in the load_dataset.
Example usage:
python HellaSwag.py

How the Evaluation Works:

1. LAMBADA:
   Dataset: LAMBADA.
   Metric: Accuracy (predicting the last word in a passage).
   Method: For each passage, the final token is removed; GPT‑2 predicts the next token, which is compared to the ground truth.
2. WikiText2:
   Dataset: WikiText2 (test split).
   Metric: Perplexity (PPL), lower is better.
   Method: Tokenize the text, compute the cross-entropy loss, then convert to perplexity via exp(average negative log-likelihood).
3. PIQA:
   Dataset: PIQA (Physical Interaction QA).
   Metric: Zero-shot multiple-choice accuracy.
   Method: Each example has a “goal” plus two solutions (sol1, sol2). Compute log-prob of (goal + sol1) vs. (goal + sol2); the higher log-prob solution is the predicted answer.
4. HellaSwag:
   Dataset: HellaSwag.
   Metric: Zero-shot multiple-choice accuracy.
   Method: Each example has a partial context plus four endings. GPT‑2 computes log-prob for each (context + ending), and the highest log-prob is chosen.

Interpreting & Comparing Results:

The original GPT‑2 paper’s numbers for these tasks can be found in Table 3 of Language Models are Unsupervised Multitask Learners. Actual results may differ because of:

Dataset Versions: The Hugging Face splits can differ from the original.
Zero‑Shot Prompting: The paper may have used slightly different prompt contexts.
Tokenization: Minor differences can affect accuracy or perplexity.
These scripts still provide a meaningful zero‑shot comparison that mirrors the original methodology, helping you see how GPT‑2 performs in your setup.

GPU vs. CPU:
If you have a compatible NVIDIA GPU, PyTorch will likely detect it automatically (torch.cuda.is_available() == True). The scripts move the model to GPU if available, greatly speeding up inference. If you see Using device: cpu, verify your CUDA installation.

Known Issues:
Symlink Warnings (Windows): You might see warnings about symlinks in the Hugging Face cache. You can ignore them or enable Developer Mode in Windows.
trust_remote_code=True: For PIQA and HellaSwag, this is required to load the dataset. Without it, you may get a ValueError about custom code.

References:
GPT‑2 Model on Hugging Face: openai-community/gpt2
Original Paper: Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
Transformers Documentation: https://huggingface.co/docs/transformers
