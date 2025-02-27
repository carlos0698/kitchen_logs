import re
import torch
import json
import os
from transformers import BertTokenizerFast, BertForTokenClassification

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Point to the root directory
MODEL_PATH = os.path.join(BASE_DIR, "scripts", "bert_kitchen_logs_model")
INPUT_FILE = os.path.join(BASE_DIR, "data", "synthetic_kitchen_logs_disordered.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "results", "output.json")

# Load trained BERT model
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

# Regular expressions to extract structured elements
ORDER_ID_PATTERN = r"Order\s+#?(\d+)"
TIME_PATTERN = r"\b(?:\d{1,2}:\d{2} (?:AM|PM))\b"

def extract_with_regex(text):
    """
    Extracts order ID and time using regex patterns.
    """
    order_id = re.search(ORDER_ID_PATTERN, text)
    time = re.search(TIME_PATTERN, text)

    return {
        "order_id": order_id.group(1) if order_id else "",
        "time": time.group(0) if time else "",
    }

# Label mapping for BERT predictions
LABEL_MAP = {
    0: "O",
    1: "B-FOOD",
    2: "I-FOOD",
    3: "B-ACTION",
    4: "I-ACTION",
}

def extract_with_bert(text):
    """
    Uses BERT to extract food items and actions.
    """
    tokens = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", padding=True, truncation=True)
    offsets = tokens.pop("offset_mapping")

    with torch.no_grad():
        outputs = model(**tokens)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    extracted_entities = {"food": [], "action": []}
    current_entity = None

    for token, label_idx, offset in zip(tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze()), predictions, offsets.squeeze()):
        label = LABEL_MAP.get(label_idx, "O")
        if label.startswith("B-"):
            current_entity = label[2:].lower()
            extracted_entities[current_entity] = [token]
        elif label.startswith("I-") and current_entity:
            extracted_entities[current_entity].append(token)
        else:
            current_entity = None

    # Convert token lists into strings
    for key in extracted_entities:
        extracted_entities[key] = tokenizer.convert_tokens_to_string(extracted_entities[key]).replace(" ##", "").strip()

    return extracted_entities

def extract_information(text):
    """
    Combines regex and BERT extraction for complete log processing.
    """
    regex_results = extract_with_regex(text)
    bert_results = extract_with_bert(text)

    return {
        "order_id": regex_results["order_id"],
        "time": regex_results["time"],
        "food": bert_results.get("food", ""),
        "action": bert_results.get("action", ""),
    }

# Process the log file
processed_logs = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            extracted_info = extract_information(line)
            processed_logs.append(extracted_info)

# Save results to JSON
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed_logs, f, indent=4)

print(f"✅ Processing complete. Results saved to '{OUTPUT_FILE}'.")
