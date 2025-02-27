import json
import random
import os

# Define base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define example lists
order_ids = [f"Order #{random.randint(1000, 9999)}" for _ in range(100)]
actions = ["loaded", "refilled", "cleaned"]
foods = ["pizza", "burger", "salad", "fries", "soda", "coffee"]
times = [f"{random.randint(1, 12)}:{random.randint(10, 59)} {'AM' if random.random() > 0.5 else 'PM'}" for _ in range(100)]

# Define sentence templates
templates = [
    "{order} {action} the {food} at {time}.",
    "At {time}, the {food} was {action} ({order}).",
    "{food} {action} - {order} - {time}",
    "{order} -> {time} -> {food} {action}.",
    "{time} | {order} | {food} got {action}."
]

# Generate synthetic dataset
dataset = []
for _ in range(500):  # Generate 500 examples
    order = random.choice(order_ids)
    action = random.choice(actions)
    food = random.choice(foods)
    time = random.choice(times)

    text = random.choice(templates).format(order=order, action=action, food=food, time=time)

    # Expected structured output
    structured_output = {
        "order_id": order.replace("Order #", ""),
        "action": action,
        "food": food,
        "time": time
    }

    dataset.append({"text": text, "output": structured_output})

# Save dataset in JSONL format using relative paths
output_dir = os.path.join(BASE_DIR, "data")
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

output_path = os.path.join(output_dir, "kitchen_logs.jsonl")
with open(output_path, "w") as f:
    for entry in dataset:
        json.dump(entry, f)
        f.write("\n")

print(f"Synthetic dataset saved to: {output_path}")