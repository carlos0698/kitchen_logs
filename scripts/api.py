import os
import re
import torch
from fastapi import FastAPI, UploadFile, File
from transformers import BertTokenizerFast, BertForTokenClassification
from tempfile import NamedTemporaryFile

# Definir rutas
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Cargar modelo entrenado
MODEL_PATH = os.path.join(BASE_DIR, "scripts", "bert_kitchen_logs_model")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

# Inicializar FastAPI
app = FastAPI()

# Expresiones regulares para extraer información
ORDER_ID_PATTERN = r"Order\s+#?(\d+)"
TIME_PATTERN = r"\b(?:\d{1,2}:\d{2} (?:AM|PM))\b"

# Mapeo de etiquetas del modelo BERT
LABEL_MAP = {
    0: "O",
    1: "B-FOOD",
    2: "I-FOOD",
    3: "B-ACTION",
    4: "I-ACTION",
}

def extract_with_regex(text):
    """Extrae order_id y time usando regex."""
    order_id = re.search(ORDER_ID_PATTERN, text)
    time = re.search(TIME_PATTERN, text)
    
    return {
        "order_id": order_id.group(1) if order_id else "",
        "time": time.group(0) if time else "",
    }

def extract_with_bert(text):
    """Usa BERT para extraer 'food' y 'action'."""
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

    # Convertir tokens en strings legibles
    for key in extracted_entities:
        extracted_entities[key] = tokenizer.convert_tokens_to_string(extracted_entities[key]).replace(" ##", "").strip()

    return extracted_entities

def extract_information(text):
    """Combina regex y BERT para extraer toda la info."""
    regex_results = extract_with_regex(text)
    bert_results = extract_with_bert(text)

    return {
        "order_id": regex_results["order_id"],
        "time": regex_results["time"],
        "food": bert_results.get("food", ""),
        "action": bert_results.get("action", ""),
    }

@app.post("/process/")
async def process_file(file: UploadFile = File(...)):
    """Recibe un archivo .txt, procesa las líneas y devuelve un JSON."""
    temp_file = NamedTemporaryFile(delete=False)
    try:
        contents = await file.read()
        temp_file.write(contents)
        temp_file.close()

        processed_logs = []
        with open(temp_file.name, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    extracted_info = extract_information(line)
                    processed_logs.append(extracted_info)

        return processed_logs

    finally:
        # Limpiar archivo temporal
        os.remove(temp_file.name)
