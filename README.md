# Kitchen Logs - BERT Model for Kitchen/Restaurant Log Standardization

## 📝 Project Overview
This project uses BERT (Bidirectional Encoder Representations from Transformers) to extract structured information from unstructured kitchen logs in a restaurant setting. The goal is to process logs (e.g., "Order #123 refilled pasta at 12:30 PM") and output key entities like order IDs, food items, actions, and times in JSON format, with a FastAPI-based API for real-time processing.

## 🔍 Approach
### Why BERT?
BERT, a pre-trained NLP model by Google, excels at understanding bidirectional context. We fine-tune it on a synthetic dataset for Named Entity Recognition (NER), adapting it efficiently to kitchen logs without training from scratch.

### Data Processing Pipeline
1. **Preprocessing with REGEX**:  
   Regular expressions extract order IDs (e.g., "Order #123" → "123") and timestamps (e.g., "12:30 PM") to lighten BERT's load.

2. **Fine-Tuning BERT for NER**:  
   We fine-tune `bert-base-uncased` on a synthetic JSONL dataset (`kitchen_logs.jsonl`) to label entities: "ORDER", "ACTION", "FOOD", "TIME". Training uses 10 epochs and AdamW optimizer.

3. **Inference with REGEX + BERT**:  
   - REGEX extracts `order_id` and `time` using patterns (`Order\s+#?\d+`, `\d{1,2}:\d{2} (?:AM|PM)`).
   - BERT identifies `food` and `action` from tokenized text.
   - Results are merged into a JSON structure.

4. **Testing**:  
   Tested on a synthetic dataset (`synthetic_kitchen_logs_disordered.txt`) to validate extraction accuracy.

5. **Deployment with FastAPI**:  
   A REST API (`api.py`) accepts `.txt` file uploads, processes logs line-by-line, and returns JSON responses in real-time.

   
```bash
kitchen_orders/
│── data/               l) Folder for datasets
│── results/            al) Folder for results
│── scripts/             # Contains all scripts
│   │── preprocessing.py  # Data preprocessing & REGEX filtering
│   │── training.py       # Fine-tuning BERT model
│   │── inference.py      # Model inference for extracting information
│   │── performance.py    # Model evaluation & performance metrics
│   │── api.py            # FastAPI server
│── requirements.txt      # Required dependencies
│── README.md            # Project documentation
```


## 📦 Using the Pre-trained BERT Model  
If you want to use the fine-tuned BERT model, download it from the following link and extract the files:  

📥 **[Download BERT Model (Google Drive)](https://drive.google.com/file/d/1KGSuMZmqdblERBOSRoYxFfQIXzLwqOV4/view?usp=share_link)**  
