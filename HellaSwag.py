import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math

def evaluate_hellaswag_accuracy(model_name="gpt2", output_file="results_hellaswag.txt"):
    """
    GPT-2 on HellaSwag (multiple-choice, zero-shot),
    using a subsequence approach to compute conditional log-probabilities
    for each [context -> ending].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # HellaSwag has custom code in its repo, so trust_remote_code=True is required
    print("Loading HellaSwag dataset (validation split)...")
    dataset = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    print(f"Loaded {len(dataset)} examples.")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    
    print("Starting HellaSwag evaluation...")


    def compute_conditional_logprob(context_str, ending_str):
        # Tokenize the context (without adding special tokens)
        context_enc = tokenizer(context_str, return_tensors="pt", add_special_tokens=False)
        
        # Tokenize the ending (prefix with a space to separate from context)
        ending_enc = tokenizer(" " + ending_str.strip(), return_tensors="pt", add_special_tokens=False)
        
        # Combine input IDs: the entire sequence is context + ending
        # but we skip the first token of `ending_enc` to avoid repeating the BOS-like token
        combined_input_ids = torch.cat(
            [context_enc["input_ids"], ending_enc["input_ids"][:, 1:]], dim=1
        ).to(device)

        with torch.no_grad():
            outputs = model(combined_input_ids)
        
        # Get the logits [sequence_length, vocab_size]
        logits = outputs.logits[0]  # shape: [seq_length, vocab_size]
        
        # The "ending" tokens start after the context ends.
        # offset = number of tokens in context_enc
        offset = context_enc["input_ids"].shape[1]

        # So these are the relevant logits for the ending portion:
        relevant_logits = logits[offset-1:-1, :]  # from offset-1 up to second-to-last row
        # And these are the actual tokens we want to match:
        relevant_targets = combined_input_ids[0, offset:]  # from offset..end

        # Convert logits to log-probs
        log_probs = torch.nn.functional.log_softmax(relevant_logits, dim=-1)
        
        # Sum log-probs for the correct tokens in the ending
        logprob_sum = log_probs[
            range(relevant_targets.size(0)), relevant_targets
        ].sum()
        
        return logprob_sum.item()

    for idx, example in enumerate(dataset):
        # Build a single context string from ctx_a and ctx_b
        ctx_a = example["ctx_a"] or ""
        ctx_b = example["ctx_b"] or ""
        context = f"{ctx_a.strip()} {ctx_b.strip()}"
        
        # The dataset provides 4 possible endings
        endings = example["endings"]
        label = example["label"]  # the correct ending index (0..3)

        # Compute conditional log-prob for each ending
        logprob_values = []
        for choice_idx in range(4):
            cond_logprob = compute_conditional_logprob(context, endings[choice_idx])
            logprob_values.append(cond_logprob)
        
        # Pick the ending with the highest conditional log-prob
        predicted_label = max(range(4), key=lambda i: logprob_values[i])
        if predicted_label == label:
            correct += 1
        total += 1

        # Optional progress print every 200 examples
        if (idx + 1) % 200 == 0:
            print(f"Processed {idx+1} / {len(dataset)} examples...")

    accuracy = 100.0 * correct / total
    print(f"\nHellaSwag Accuracy for {model_name}: {accuracy:.2f}% (total: {total} examples)")

    # Write results to a file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== HellaSwag Evaluation Results ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Total Examples: {total}\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    evaluate_hellaswag_accuracy("gpt2", "results_hellaswag.txt")
