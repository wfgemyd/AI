import torch
import math
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def compute_perplexity(
    model_name="gpt2",
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    output_file="results_wikitext2.txt"
):
    """
    GPT-2 perplexity on a given dataset (default: WikiText2)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load dataset (test split)
    print(f"Loading dataset: {dataset_name}/{dataset_config} (test split)...")
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    print(f"Loaded {len(dataset)} examples.")

    # 2. Load tokenizer & model, move model to GPU
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Some models (like GPT-2) do not have a pad token; set it if needed
    tokenizer.pad_token = tokenizer.eos_token

    total_log_likelihood = 0.0
    total_tokens = 0

    print("\nStarting perplexity evaluation on WikiText2...")
    for idx, example in enumerate(dataset):
        text = example["text"]
        if not text.strip():
            continue
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # outputs.logits shape: [batch_size, seq_length, vocab_size]
        # We want log probs of each correct token
        logits = outputs.logits[:, :-1, :].contiguous()   # all but last token
        target_ids = input_ids[:, 1:].contiguous()        # shift by 1

        # Cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        total_log_likelihood += loss.item()
        total_tokens += target_ids.numel()

        # Optional: print progress every 500 examples
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx+1} / {len(dataset)} examples...")

    if total_tokens == 0:
        print("No valid tokens found!")
        return

    # Perplexity = exp(mean negative log-likelihood)
    ppl = math.exp(total_log_likelihood / total_tokens)

    # Print final result
    print(f"\nPerplexity on {dataset_name}/{dataset_config} for {model_name}: {ppl:.2f}")

    # Save to text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== WikiText2 Perplexity Results ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Perplexity: {ppl:.2f}\n")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    compute_perplexity(
        model_name="gpt2",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        output_file="results_wikitext2.txt"
    )
