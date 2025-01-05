import torch
import math
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def evaluate_piqa_accuracy(model_name="gpt2", output_file="results_piqa.txt"):
    """
    GPT-2 on PIQA (zero-shot multiple-choice).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Add `trust_remote_code=True` to load_dataset
    print("Loading PIQA dataset (validation split)...")
    dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
    print(f"Loaded {len(dataset)} examples.")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    print("Starting PIQA evaluation...")
    for idx, example in enumerate(dataset):
        goal = example["goal"]
        sol1 = example["sol1"]
        sol2 = example["sol2"]
        label = example["label"]  # 0 or 1

        # We'll compute log-likelihood for [goal + sol1] vs. [goal + sol2]
        def compute_logprob(text):
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"])
            logits = outputs.logits[:, :-1, :]
            target_ids = inputs["input_ids"][:, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            logits_flat = log_probs.view(-1, log_probs.size(-1))
            targets_flat = target_ids.view(-1)
            return logits_flat[range(logits_flat.size(0)), targets_flat].sum()

        text_sol1 = f"{goal.strip()} {sol1.strip()}"
        text_sol2 = f"{goal.strip()} {sol2.strip()}"

        logprob_sol1 = compute_logprob(text_sol1)
        logprob_sol2 = compute_logprob(text_sol2)

        pred_label = 0 if logprob_sol1 > logprob_sol2 else 1
        if pred_label == label:
            correct += 1
        total += 1

        if (idx + 1) % 200 == 0:
            print(f"Processed {idx+1} / {len(dataset)} examples...")

    accuracy = 100.0 * correct / total
    print(f"\nPIQA Accuracy for {model_name}: {accuracy:.2f}% (total: {total} examples)")

    # Write results to a text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== PIQA Evaluation Results ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Total Examples: {total}\n")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    evaluate_piqa_accuracy("gpt2", "results_piqa.txt")
