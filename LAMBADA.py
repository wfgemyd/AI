import torch
import math
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def evaluate_lambada_accuracy(model_name="gpt2", output_file="results_lambada.txt"):
    """
    GPT-2 on the LAMBADA dataset for accuracy 
    """
    # 1. Detect GPU or fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading LAMBADA dataset...")
    dataset = load_dataset("lambada")["validation"]
    print(f"LAMBADA dataset length: {len(dataset)} examples.")

    # 2. Load tokenizer & model, then move model to GPU
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    correct_predictions = 0
    total = 0

    print("Starting evaluation... This can take time, but should be faster on GPU.\n")
    for idx, example in enumerate(dataset):
        text = example["text"]
        # Tokenize the entire passage
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Skip if not enough tokens to evaluate
        if len(tokens) < 2:
            continue

        # Last token is what we want to predict
        context_tokens = tokens[:-1]
        last_token = tokens[-1]
        
        total += 1
        
        # Prepare input for the model, move to GPU
        input_ids = torch.tensor([context_tokens]).to(device)

        # Forward pass (no gradient needed)
        with torch.no_grad():
            outputs = model(input_ids)

        # The final logits for the next token are at the last position
        next_token_logits = outputs.logits[0, -1, :]
        predicted_token = torch.argmax(next_token_logits).item()
        
        if predicted_token == last_token:
            correct_predictions += 1

        # Optional: progress print every 500 examples
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx+1} / {len(dataset)} examples...")

    if total == 0:
        print("No valid samples found. Check dataset loading!")
        return

    accuracy = (correct_predictions / total) * 100
    print(f"\nEvaluation complete!\nLAMBADA Accuracy for {model_name}: {accuracy:.2f}%  (total: {total} examples)\n")

    # 3. Write results to a text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== LAMBADA Evaluation Results ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Total Examples: {total}\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    evaluate_lambada_accuracy("gpt2", "results_lambada.txt")
