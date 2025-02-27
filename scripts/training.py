import json
import torch
import os
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset

# Define base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ✅ 1. Load synthetic dataset
data = []
dataset_path = os.path.join(BASE_DIR, "data", "kitchen_logs.jsonl")
with open(dataset_path, "r") as f:
    for line in f:
        data.append(json.loads(line))

# ✅ 2. Tokenize text and assign labels
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Labels for NER task
labels_map = {"O": 0, "B-ORDER": 1, "I-ORDER": 2, "B-ACTION": 3, "I-ACTION": 4, "B-FOOD": 5, "I-FOOD": 6, "B-TIME": 7, "I-TIME": 8}

def tokenize_and_label(example):
    tokens = tokenizer(example["text"], padding="max_length", truncation=True, return_offsets_mapping=True)

    # Generate token labels
    labels = ["O"] * len(tokens["input_ids"])  # Default: Outside (O)
    text = example["text"]

    # Assign labels based on expected output
    for key, value in example["output"].items():
        if key == "order_id":
            label_type = "B-ORDER"
        elif key == "action":
            label_type = "B-ACTION"
        elif key == "food":
            label_type = "B-FOOD"
        elif key == "time":
            label_type = "B-TIME"
        else:
            continue

        idx = text.find(str(value))
        if idx != -1:
            start_token = tokens.char_to_token(idx)
            if start_token is not None:
                labels[start_token] = label_type
                for i in range(start_token + 1, len(tokens["input_ids"])):
                    if tokens.char_to_token(i) is None:
                        break
                    labels[i] = "I-" + label_type.split("-")[1]

    tokens["labels"] = [labels_map[label] for label in labels]
    return tokens

# Apply tokenization and labeling
dataset = Dataset.from_list(data)
dataset = dataset.map(tokenize_and_label, remove_columns=["text", "output"])

# ✅ 3. Define BERT model for token classification (NER)
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(labels_map))

# ✅ 4. Configure training parameters
training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "bert_kitchen_logs"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(BASE_DIR, "logs"),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer
)

# ✅ 5. Train the model
trainer.train()

# ✅ 6. Save the model
model_dir = os.path.join(BASE_DIR, "scripts","bert_kitchen_logs_model")
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print("✅ BERT fine-tuning completed and model saved.")