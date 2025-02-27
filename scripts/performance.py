import os
import json
from sklearn.metrics import precision_recall_fscore_support

# Define paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Get current script directory
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "data", "test_labels.json")
PREDICTED_PATH = os.path.join(BASE_DIR, "results", "output.json")

# Load ground truth data
with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)

# Load predicted data
with open(PREDICTED_PATH, "r", encoding="utf-8") as f:
    predicted_data = json.load(f)

# Extract true and predicted labels for each category
true_order_ids, pred_order_ids = [], []
true_actions, pred_actions = [], []
true_food_items, pred_food_items = [], []
true_times, pred_times = [], []

for gt, pred in zip(ground_truth_data, predicted_data):
    true_order_ids.append(str(gt["order_id"]))  
    pred_order_ids.append(str(pred["order_id"]))

    true_actions.append(gt["action"])
    pred_actions.append(pred["action"])

    true_food_items.append(gt["food_item"])
    pred_food_items.append(pred["food"]) 

    true_times.append(gt["time"])
    pred_times.append(pred["time"])

# Compute precision, recall, and F1-score for each category
metrics = {}
for category, true_vals, pred_vals in zip(
    ["order_id", "action", "food_item", "time"],
    [true_order_ids, true_actions, true_food_items, true_times],
    [pred_order_ids, pred_actions, pred_food_items, pred_times]
):
    precision, recall, f1, _ = precision_recall_fscore_support(true_vals, pred_vals, average="weighted")
    metrics[category] = {"precision": precision, "recall": recall, "f1": f1}

# Save results to JSON
OUTPUT_METRICS_PATH = os.path.join(BASE_DIR, "results", "evaluation_metrics.json")
os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)

with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# Print results
print(f"✅ Evaluation completed. Metrics saved to '{OUTPUT_METRICS_PATH}'.")
print(json.dumps(metrics, indent=4))
