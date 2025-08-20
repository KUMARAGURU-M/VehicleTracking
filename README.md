🚗 Real Time Vehicle Tracking System using AI & Multimodal Data

  This project presents an end-to-end AI-based Vehicle Tracking System designed for real-time surveillance and intelligent monitoring of vehicles across a distributed camera network. By leveraging computer vision, deep learning, multimodal data fusion, and graph traversal algorithms, the system efficiently identifies and tracks vehicles using parameters like license plate, make, model, and color—even across multiple camera nodes.
  

📌 Problem Statement

  In urban environments, tracking vehicles across a network of surveillance cameras is a critical task for law enforcement, traffic analysis, and security. Manual analysis of camera feeds is inefficient and error-prone. This project automates the vehicle identification and tracking process using AI-driven object detection and multimodal matching, enabling real-time and retrospective tracking across large-scale camera networks.


🚀 Project Highlights

🎯 1. YOLO-based Vehicle Detection

  Utilizes YOLOv8 for high-speed, high-accuracy vehicle detection in video feeds.
  Captures vehicle bounding boxes for further processing like plate recognition and re-identification.
  

🔢 2. License Plate Recognition (LPR)

  Detects and extracts license plate regions using another YOLO model.
  Applies OCR using EasyOCR to extract alphanumeric license text from cropped plate regions.
  Supports confidence scoring and visual annotations.
  

🧠 3. Multimodal Vehicle Identification

  Matches vehicles using multiple identification cues:
  License plate text
  Vehicle model type
  Company (manufacturer)
  Color
  Fallback strategy: even if license plate OCR fails, other modalities help in identifying the vehicle.
  

📡 4. Camera Graph & Path Traversal

  Each surveillance camera is treated as a node in a connected graph.
  Uses Breadth-First Search (BFS) for wide-area scanning and Depth-First Search (DFS) for continuous chain detection.
  Ensures efficient traversal through unvisited camera nodes while avoiding redundancy.
  

🔄 5. Temporal Tracking with LSTM (Optional Module)

  Incorporates LSTM (Long Short-Term Memory) models to predict vehicle trajectory and suggest next probable camera nodes.
  Useful for estimating the vehicle’s path if real-time feed is unavailable or occluded.
  

🧾 6. MongoDB Integration with Vector Search

  Stores known vehicle records with metadata.
  Matches detected vehicle info against the database for real-time validation.
  Supports partial and fuzzy matching across multiple fields.
  

🛠 Tech Stack

  Component	Technology Used
  Programming	Python
  Object Detection	YOLO11 (via Ultralytics)
  OCR	PaddleOCR
  Deep Learning	PyTorch, TensorFlow (LSTM)
  Data Handling	OpenCV, NumPy
  Database	MongoDB with vector-based indexing
  Graph Traversal	NetworkX
  Interface	CLI + placeholder for future Web UI


📈 Workflow Summary

  User inputs search parameters (e.g., license plate, model).
  System identifies starting camera and initiates detection.
  Vehicle is detected using YOLO → license plate extracted & OCR applied.
  If matched with user input, traversal continues to connected cameras.
  MongoDB is queried with structured data to confirm matches.
  Final output: cameras where the vehicle was found + tracking path.


✅ Use Cases

  Vehicle tracing for law enforcement and smart policing.
  Stolen vehicle recovery and alert systems.
  Smart city traffic analytics and congestion mapping.
  Automated tolling, parking, and checkpoint validation.


📁 Directory Overview

├── models/                # YOLO models for vehicle & plate detection

├── tracker/               # Core logic: BFS, DFS, detection pipeline

├── db/                    # MongoDB interfacing & vector search utils

├── lstm_module/           # Predictive modeling using LSTM

├── frontend/            

├── utils/                 # Helper functions (color mapping, preprocessing)

├── main.py                # Entry point for vehicle tracking

└── README.md              # This file


👨‍💻 Author :
  Kumaraguru M Suriya SK Surendar B Gnaneshwaran JS

This project was developed as part of an AI/ML capstone focused on real-world applications of multimodal recognition and deep learning. It showcases how AI can interact with databases, detect and track physical entities across spatial networks, and generate explainable outputs.
