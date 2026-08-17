# data.py
import tiktoken
import torch
from datasets import load_dataset

def load_and_tokenize_data(sample_limit=10000):
    print(f"Loading {sample_limit} stories from TinyStories on Hugging Face...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Collect a subset of stories for fast training
    text_corpus = []
    for i, item in enumerate(dataset):
        text_corpus.append(item["text"])
        if i + 1 >= sample_limit:
            break
            
    full_text = "\n<|endoftext|>\n".join(text_corpus)
    tokens = enc.encode(full_text, allowed_special={"<|endoftext|>"})
    
    data_tensor = torch.tensor(tokens, dtype=torch.long)
    n = int(0.9 * len(data_tensor))
    train_data = data_tensor[:n]
    val_data = data_tensor[n:]
    
    print(f"Tokenized {len(data_tensor):,} total tokens across {sample_limit} stories.")
    return train_data, val_data, enc

if __name__ == "__main__":
    load_and_tokenize_data(sample_limit=1000)