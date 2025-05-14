import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# ----------------------------
# 1. Define Balanced Transitions
# ----------------------------
balanced_transitions = {
    (1, 1, 3): 25000,   # 1 → 3
    (1, 1, 2): 12000,   # 1 → 2
    (3, 3, 7): 25000,
    (1, 3, 7): 27000,   # 3 → 7
    (3, 7, 6): 35000,   # 7 → 6
    (7, 7, 6): 15000,
    (3, 7, 8): 12000,   # 7 → 8
    
    # Node 2 transitions
    (2, 2, 5): 18000,   # 2 → 5 (primary)
    (2, 2, 1): 15000,   # 2 → 1
    (2, 2, 4): 15000,   # 2 → 4
    
    # Node 4 transitions
    (4, 4, 2): 15000,   # 4 → 2
    (4, 4, 3): 15000,   # 4 → 3
    (4, 4, 8): 15000,   # 4 → 8
    
    # Node 5 transitions
    (5, 5, 2): 18000,   # 5 → 2
    (2, 5, 2): 10000,   # Return path
    
    # Node 8 transitions
    (8, 8, 4): 15000,   # 8 → 4
    (8, 8, 7): 15000,   # 8 → 7
    
    # Cross-node transitions
    (4, 3, 4): 12000,   # 3 → 4
    (5, 2, 4): 15000,   # 2 → 4
    (5, 2, 1): 15000,   # 2 → 1
    (2, 4, 8): 15000,   # 4 → 8
    (2, 4, 3): 15000,   # 4 → 3
    (4, 8, 7): 15000    # 8 → 7
    
}
# # Additional variety transitions
# additional_transitions = {
#     (2, 1, 5): 1000,   
#     (4, 3, 8): 1000    
# }

# ----------------------------
# 2. Define Direction Mapping
# ----------------------------
direction_mapping = {
    (1, 1, 3): "East",
    (1, 1, 2): "West",
    (1, 3, 7): "South",
    (3, 3, 7): "South",
    (4, 3, 4): "North",
    (3, 7, 6): "East",
    (3, 7, 8): "West",
    (7, 7, 6): "West",
    (2, 1, 5): "Southt",
    (4, 3, 8): "North",


    (2, 2, 5): "South", 
    (2, 2, 1): "North",
    (2, 2, 4): "North",
    (4, 4, 2): "South",
    (4, 4, 3): "South",
    (4, 4, 8): "East",
    (5, 5, 2): "North",
    (8, 8, 4): "West",
    (8, 8, 7): "South",
    (5, 2, 4): "North",
    (5, 2, 1): "North",
    (2, 4, 8): "East",
    (2, 4, 3): "South",
    (4, 8, 7): "South"
}

possible_directions = ["North", "South", "East", "West"]

# ----------------------------
# 3. Generate Balanced Transitions
# ----------------------------
def generate_transitions(prev_node, current_node, next_node, count, start_time):
    time_increment = timedelta(seconds=5)
    current_time = start_time
    data = []
    
    for _ in range(count):
        # More realistic speed distribution (Normal dist: Mean 50, StdDev 10, clipped to [20, 80])
        speed = max(20, min(80, np.random.normal(50, 10)))

        # Assign direction dynamically if not in mapping
        direction = direction_mapping.get((prev_node, current_node, next_node), random.choice(possible_directions))

        data.append([
            current_time.strftime("%H:%M:%S"),
            prev_node,
            current_node,
            next_node,
            direction,
            round(speed, 1)  # Round to 1 decimal place
        ])
        current_time += time_increment
    return data, current_time

# ----------------------------
# 4. Create the Balanced Dataset
# ----------------------------
dataset = []
base_time = datetime.strptime("08:00:00", "%H:%M:%S")

# Generate balanced transitions
for key, count in balanced_transitions.items():
    prev_node, current_node, next_node = key
    data_batch, base_time = generate_transitions(prev_node, current_node, next_node, count, base_time)
    dataset.extend(data_batch)

# Generate additional transitions
# for key, count in additional_transitions.items():
#     prev_node, current_node, next_node = key
#     data_batch, base_time = generate_transitions(prev_node, current_node, next_node, count, base_time)
#     dataset.extend(data_batch)

# ----------------------------
# 5. Save to CSV
# ----------------------------
df = pd.DataFrame(dataset, columns=["Time", "Previous Node", "Current Node", "Predicted Next Node", "Direction", "Speed (km/h)"])
dataset_path = "balanced_route_prediction_dataset.csv"
df.to_csv(dataset_path, index=False)
print(f"✅ Balanced dataset generated with {len(df)} records and saved as '{dataset_path}'.")
