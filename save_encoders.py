# save_encoders.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib

# Load your dataset
csv_path = "balanced_route_prediction_dataset.csv"
df = pd.read_csv(csv_path)

# 1. Create and save direction encoder
direction_encoder = OneHotEncoder(sparse_output=False)
direction_encoder.fit(df[["Previous Node", "Current Node", "Direction"]])
joblib.dump(direction_encoder, "direction_encoder.joblib")

# 2. Create and save target encoder
target_encoder = OneHotEncoder(sparse_output=False)
target_encoder.fit(df[["Predicted Next Node"]])
joblib.dump(target_encoder, "target_encoder.joblib")

# 3. Create and save speed scaler
speed_scaler = MinMaxScaler()
speed_scaler.fit(df[["Speed (km/h)"]])
joblib.dump(speed_scaler, "speed_scaler.joblib")

print("Encoders and scaler saved successfully!")