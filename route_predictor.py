import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer as BaseInputLayer # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import joblib

# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class CustomInputLayer(BaseInputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['shape'] = kwargs.pop('batch_shape')[1:]
        super().__init__(**kwargs)

# Load model with silent warnings
tf.get_logger().setLevel('ERROR')
model = load_model(
    r"E:\Unisys3\final_route_prediction_model.h5",
    custom_objects={'InputLayer': CustomInputLayer}
)

# Silent compilation (use your actual training config)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[]  # Empty metrics for inference-only use
)

# Load encoders with validation
encoder = joblib.load("direction_encoder.joblib")
target_encoder = joblib.load("target_encoder.joblib")
scaler = joblib.load("speed_scaler.joblib")

# Constants from training
NODE_FEATURES = ["Previous Node", "Current Node", "Direction"]
SPEED_FEATURE = ["Speed (km/h)"]
SEQUENCE_LENGTH = 5

def preprocess_input(prev_node, curr_node, speed, direction):
    """Robust preprocessing with full validation"""
    try:
        # Type enforcement
        prev_node = int(prev_node)
        curr_node = int(curr_node)
        speed = float(speed)
        
        # Handle initial state using training convention
        if prev_node == curr_node:
            direction = "East"
            
        # Validate direction against encoder
        valid_directions = list(encoder.categories_[2])
        direction = direction if direction in valid_directions else valid_directions[0]
        
        # Create validated DataFrames
        node_data = pd.DataFrame([[prev_node, curr_node, direction]], 
                                columns=NODE_FEATURES)
        speed_data = pd.DataFrame([[speed]], columns=SPEED_FEATURE)
        
        # Transform features
        encoded_nodes = encoder.transform(node_data)
        normalized_speed = scaler.transform(speed_data)[0][0]
        
        # Build sequence
        timestamp = 8 * 3600  # 8AM
        sequence = np.tile(
            np.concatenate([[timestamp, normalized_speed], encoded_nodes[0]]),
            (SEQUENCE_LENGTH, 1)
        )
        
        return np.expand_dims(sequence, axis=0)
        
    except Exception as e:
        pass
        # print(f"Preprocessing error: {str(e)}")
        return None

def predict_next_node(previous_node, current_node, speed, direction):
    """Production-ready prediction function"""
    try:
        processed = preprocess_input(previous_node, current_node, speed, direction)
        if processed is None:
            return None
            
        preds = model.predict(processed, verbose=0)
        return int(target_encoder.categories_[0][np.argmax(preds[0])])
        
    except Exception as e:
        pass
        # print(f"Prediction failed: {str(e)}")
        return None
# # Test initial state prediction
# print(predict_next_node(7, 7, 45, "west"))  # Should return 3
# # Test unknown direction
# print(predict_next_node(3, 7, 50, "South"))  # Uses valid direction fallback