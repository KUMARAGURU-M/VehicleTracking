import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from datetime import datetime

# ----------------------------
# 1. Load the Dataset
# ----------------------------
csv_path = "balanced_route_prediction_dataset.csv"
df = pd.read_csv(csv_path)
print("✅ Raw Dataset Loaded!")
print(df.head())

# ----------------------------
# 2. Convert Time to Numeric (Seconds)
# ----------------------------
def time_to_seconds(time_str):
    t = datetime.strptime(time_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second

df["Time_seconds"] = df["Time"].apply(time_to_seconds)

# ----------------------------
# 3. Normalize the Speed (km/h)
# ----------------------------
scaler = MinMaxScaler()
df["Speed_normalized"] = scaler.fit_transform(df[["Speed (km/h)"]])

# ----------------------------
# 4. One-Hot Encode Categorical Features
# ----------------------------
categorical_cols = ["Previous Node", "Current Node", "Direction"]
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_cols])
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# ----------------------------
# 5. Combine All Features
# ----------------------------
df_features = pd.concat([df[["Time_seconds", "Speed_normalized"]], df_encoded], axis=1)

# ----------------------------
# 6. One-Hot Encode the Target "Predicted Next Node"
# ----------------------------
target_encoder = OneHotEncoder(sparse_output=False)
y_encoded = target_encoder.fit_transform(df[["Predicted Next Node"]])

# ----------------------------
# 7. Create Sequences for LSTM Training
# ----------------------------
SEQUENCE_LENGTH = 5  # Define sequence length

X_all = df_features.values
y_all = y_encoded

X_seq, y_seq = [], []
for i in range(len(X_all) - SEQUENCE_LENGTH):
    X_seq.append(X_all[i:i+SEQUENCE_LENGTH])
    y_seq.append(y_all[i+SEQUENCE_LENGTH])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("📊 X_seq shape:", X_seq.shape)  # (num_sequences, SEQUENCE_LENGTH, num_features)
print("📊 y_seq shape:", y_seq.shape)  # (num_sequences, num_classes)

# ----------------------------
# 8. Save Processed Data
# ----------------------------
np.save("X.npy", X_seq)
np.save("y.npy", y_seq)
print("✅ Preprocessing complete. Files 'X.npy' and 'y.npy' saved!")
