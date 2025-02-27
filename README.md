Kitchen Logs - BERT Model for Kitchen/Restaurant Log Standardization

📝 Project Overview

This project leverages BERT (Bidirectional Encoder Representations from 
Transformers) to detect and standardize logs in a kitchen or restaurant 
environment. The goal is to make kitchen logs and commands more uniform 
and structured.

🔍 Why BERT?
BERT is a powerful pre-trained NLP model developed by Google. Instead of 
training a model from scratch, we perform fine-tuning on a synthetic 
dataset tailored for kitchen log detection. This approach optimizes 
computational resources while effectively adapting BERT to our specific 
task.

📂 Data Processing Pipeline

1️⃣ Preprocessing with REGEX:

We use Regular Expressions (REGEX) to filter and extract basic information 
(e.g., order IDs, timestamps) before feeding data into BERT.
This step improves efficiency by reducing unnecessary processing.

2️⃣ Fine-Tuning BERT on Synthetic Data:

A synthetic dataset was created to train BERT, simulating real kitchen 
logs.
The fine-tuning process helps BERT recognize food items, actions, and 
commands commonly used in a restaurant setting.

3️⃣ Testing with Synthetic Data:

Another synthetic dataset was generated to test model performance and 
accuracy.
This ensures the model properly extracts and structures kitchen log 
information.

4️⃣ Deploying the Model with FastAPI:

A FastAPI-based REST API was implemented to provide an easy-to-use 
interface for processing kitchen logs.
Users can upload .txt files, and the API returns a structured JSON output 
with extracted information.

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
