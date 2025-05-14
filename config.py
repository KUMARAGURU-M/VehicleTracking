# config.py

# Hugging Face Token (replace with your actual token)
import os
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# MongoDB Configuration
MONGO_URI = "mongodb+srv://AI_agent:z8W1L0n41kZvseDw@unisys.t75li.mongodb.net/?retryWrites=true&w=majority&appName=Unisys"
DATABASE_NAME = "vehicle_detection"
COLLECTION_NAME = "detected_vehicles"