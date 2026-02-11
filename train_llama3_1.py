import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
import json
import os

# Paths
train_path = "Dataset/train.json"
val_path = "Dataset/val.json"
model_name = "meta-llama/Llama-3.1-8B"

# Load data
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) if line.strip().startswith('{') else json.loads(f'{{{line.strip()}}}') for line in f if line.strip()]
    return data

train_data = load_data(train_path)
val_data = load_data(val_path)

# Prepare dataset for Llama
class LlamaDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        label = str(item.get("label_id", ""))
        prompt = f"{text}\nLabel: "
        full_text = prompt + label
        encoding = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        labels = input_ids.clone()
        # Mask prompt tokens in labels
        prompt_encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        prompt_len = (prompt_encoding.input_ids.squeeze() != self.tokenizer.pad_token_id).sum().item()
        labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
# Set pad_token to eos_token for Llama
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(model_name)

dataset = LlamaDataset(train_data, tokenizer)
val_dataset = LlamaDataset(val_data, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="llama3.1_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

# Save final model
trainer.save_model("llama3.1_output/final_model")
