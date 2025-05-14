import cv2
import time 
import torch
import logging
import numpy as np
import re  
from sentence_transformers import SentenceTransformer
from Levenshtein import ratio as levenshtein_ratio
from paddleocr import PaddleOCR
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pymongo import MongoClient
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MongoDB configuration
MONGO_URI = "mongodb+srv://AI_agent:z8W1L0n41kZvseDw@unisys.t75li.mongodb.net/?retryWrites=true&w=majority&appName=Unisys"
client = MongoClient(MONGO_URI)
db = client["vehicle_detection"]
collection = db["detected_vehicles"] 

# Configure logging
logging.getLogger("ppocr").setLevel(logging.ERROR) 
logging.getLogger("ppocr").propagate = False   

# Constants
Vehicle_classes = ['Bus', 'Car', 'Two wheeler']
Number_plate_classes = ['Number Plate']
FRAME_SKIP = 3
OCR_CACHE_TIME = 2
PLATE_REGEX = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{2}\d{4}$')
PLATE_SIMILARITY_THRESHOLD = 0.8

# Color detection ranges
COLOR_RANGES = {
    'dark blue': ([100, 100, 50], [130, 255, 255]),
    'white': ([0, 0, 200], [179, 30, 255]),
    'maroon': ([0, 100, 50], [10, 255, 150], [170, 100, 50], [179, 255, 150]),
    'red': ([0, 100, 50], [10, 255, 255]),
    'orange': ([10, 100, 50], [25, 255, 255]),
    'purple': ([130, 50, 50], [170, 255, 255]),
    'pink': ([140, 50, 50], [170, 255, 255]),
    'brown': ([10, 100, 50], [20, 255, 150]),
    'cyan': ([80, 100, 50], [100, 255, 255]),
    'lime': ([35, 100, 50], [75, 255, 255]),
    'violet': ([130, 50, 50], [160, 255, 255]),
}

# License plate configuration
TARGET_PLATES = {
    "PY01BV2662": {
        "1": {"speed": "50 km/h", "lane": "2", "direction": "East"},
        "3": {"speed": "30 km/h", "lane": "2", "direction": "South"},
        "6": {"speed": "20 km/h", "lane": "2", "direction": "East"},
        "7": {"speed": "30 km/h", "lane": "2", "direction": "west"}
    },
    "TN88J9744": {
        "1": {"speed": "45 km/h", "lane": "2", "direction": "East"},
        "3": {"speed": "30 km/h", "lane": "2", "direction": "South"},
        "6": {"speed": "40 km/h", "lane": "2", "direction": "East"},
        "7": {"speed": "30 km/h", "lane": "2", "direction": "west"}
    }
}

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())

def normalize_plate(plate):
    """Normalize plate text for better matching"""
    replacements = {
        "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B",
        " ": "", "-": "", ".": "", "_": "", "I": "J", "J": "J"
    }
    return "".join([replacements.get(c.upper(), c.upper()) for c in plate 
                  if c.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"])

def find_matching_plate(detected_plate):
    """Find the best matching plate using fuzzy logic"""
    if not detected_plate or detected_plate == "Not detected":
        return None
    
    cleaned_plate = normalize_plate(detected_plate)
    
    best_match = None
    highest_similarity = 0
    
    for target_plate in TARGET_PLATES:
        target_cleaned = normalize_plate(target_plate)
        similarity = levenshtein_ratio(target_cleaned, cleaned_plate)
        
        if similarity > highest_similarity and similarity >= PLATE_SIMILARITY_THRESHOLD:
            highest_similarity = similarity
            best_match = target_plate
    
    return best_match

def save_to_mongodb(track_id, data, camera_number):
    """Save vehicle information to MongoDB with unique entries"""
    try:
        # Generate unique composite key using timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        composite_key = f"{camera_number}{track_id}{timestamp}"

        # Generate text representation
        text_representation = (
            f"Vehicle Type: {data.get('vehicle_class', 'Unknown')}, "
            f"Color: {data.get('color', 'Unknown')}, "
            f"License Plate: {data.get('plate_text', 'Not detected')}."
        )

        # Generate embeddings
        embedding = embedding_model.encode(text_representation).tolist()

        document = {
            "_id": composite_key,  # Unique identifier for each entry
            "camera_number": camera_number,
            "track_id": track_id,
            "vehicle_class": data.get("vehicle_class", "Unknown"),
            "color": data.get("color", "Unknown"),
            "license_plate": data.get("plate_text", "Not detected"),
            "timestamp": datetime.now().isoformat(),
            "text_representation": text_representation,
            "embedding": embedding
        }

        # Fuzzy plate matching
        detected_plate = data.get('plate_text', '')
        matched_plate = find_matching_plate(detected_plate)
        
        if matched_plate:
            camera_params = TARGET_PLATES[matched_plate].get(str(camera_number), {})
            document.update(camera_params)

        # Insert as new document instead of updating
        collection.insert_one(document)

    except Exception as e:
        print(f"Database error: {e}")

def detect_vehicle_color(roi):
    """Detect dominant vehicle color using HSV mask analysis"""
    try:
        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
        
        color_counts = {}
        for color, ranges in COLOR_RANGES.items():
            if len(ranges) == 4:
                lower1, upper1, lower2, upper2 = ranges
                mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                count = cv2.countNonZero(mask1) + cv2.countNonZero(mask2)
            else:
                lower, upper = ranges
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                count = cv2.countNonZero(mask)
            color_counts[color] = count
        
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            return 'unknown', 0.0
        
        color_percentages = {color: count / total_pixels for color, count in color_counts.items()}
        best_color = max(color_percentages, key=color_percentages.get)
        return best_color, color_percentages[best_color]
    except Exception as e:
        print(f"Color detection error: {e}")
        return 'unknown', 0.0

# OCR optimization cache
LAST_PLATE_TEXT = {}
PLATE_HISTORY = {}

def optimize_ocr(plate_roi, track_id):
    global LAST_PLATE_TEXT, PLATE_HISTORY
    
    current_time = time.time()
    
    if track_id in LAST_PLATE_TEXT:
        time_diff = current_time - LAST_PLATE_TEXT[track_id]
        if time_diff < OCR_CACHE_TIME:
            return PLATE_HISTORY[track_id]
    
    result = ocr_reader.ocr(plate_roi, cls=True)
    plate_text = result[0][0][1][0] if result and len(result[0]) > 0 else "Not detected"
    
    LAST_PLATE_TEXT[track_id] = current_time
    PLATE_HISTORY[track_id] = plate_text
    
    return plate_text

def load_model(path):
    model = YOLO(path).to(device)
    model.fuse()
    if device == 'cuda':
        model.half()
    return model

# Load YOLO models
vehicle_model = load_model(r"E:\Unisys3\Vehicle\Latest\best.pt")
number_plate_model = load_model(r"E:\Unisys3\Number plate\Latest\best.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=50, n_init=2)
vehicle_info = {}

def validate_roi(roi):
    return roi is not None and roi.size > 0 and roi.shape[0] > 10 and roi.shape[1] > 10
def process_video_with_yolo(video_path, camera_number):
    global vehicle_info
    vehicle_info = {}  

    frame_count = 0
    
    cv2.namedWindow(f'Camera {camera_number} - Vehicle Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'Camera {camera_number} - Vehicle Tracking', 960, 540)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    output_path = f"{camera_number}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 5
        if frame_count % (FRAME_SKIP + 5) != 0:
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Vehicle detection
        vehicle_results = vehicle_model.predict(frame, conf=0.8, verbose=False, device=device)
        
        detections = []
        for result in vehicle_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))
        
        # Object tracking
        tracked_objects = tracker.update_tracks(detections, frame=frame)
        
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Update vehicle info
            if track_id not in vehicle_info:
                class_id = int(track.det_class) if track.det_class is not None else 0
                conf = float(track.det_conf) if track.det_conf is not None else 0.0
                vehicle_info[track_id] = {

                    'vehicle_class': Vehicle_classes[class_id],
                    'vehicle_confidence': conf,
                    'color': "Unknown",
                    'color_confidence': 0.0,
                    'plate_confidence': 0.0,
                    'plate_text': "Not detected",
                    'plate_valid': False,
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now(),
                    'last_save': datetime.min  # Initialize last_save
                }
            else:
                vehicle_info[track_id]['last_seen'] = datetime.now()
            
            info = vehicle_info[track_id]
            
            # Draw vehicle bounding box
            label = f"ID:{track_id} {info['vehicle_class']} {info['vehicle_confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Color detection
            vehicle_roi = frame[y1:y2, x1:x2]
            if validate_roi(vehicle_roi):
                color, color_conf = detect_vehicle_color(vehicle_roi)
                info['color'] = color
                info['color_confidence'] = color_conf
                cv2.putText(frame, f"Color: {color} ({color_conf:.2f})", 
                          (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # License plate processing
            try:
                plate_results = number_plate_model.predict(vehicle_roi, conf=0.6, verbose=False, device=device)
                for plate in plate_results:
                    for pbox, pcls, pconf in zip(plate.boxes.xyxy.cpu().numpy(),
                                               plate.boxes.cls.cpu().numpy().astype(int),
                                               plate.boxes.conf.cpu().numpy()):
                        px1, py1, px2, py2 = map(int, pbox)
                        info['plate_confidence'] = float(pconf)
                        
                        plate_roi = vehicle_roi[py1:py2, px1:px2]
                        if validate_roi(plate_roi):
                            gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                            plate_text = optimize_ocr(gray_plate, track_id)
                            
                            plate_text = plate_text.strip().upper()
                            info['plate_text'] = plate_text
                            info['plate_valid'] = PLATE_REGEX.match(plate_text) is not None

                            # Draw plate info
                            plate_color = (0, 255, 0) if info['plate_valid'] else (0, 0, 255)
                            status_text = "Valid" if info['plate_valid'] else "Invalid"
                            cv2.rectangle(frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), plate_color, 2)
                            cv2.putText(frame, f"{info['plate_text']} ({status_text})", 
                                      (x1 + px1, y1 + py2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, plate_color, 2)
            except Exception as e:
                pass

            # Time-based saving logic
            if info['plate_text'] != "Not detected":
                # Save only once every 2 seconds per vehicle
                if (datetime.now() - info['last_save']).total_seconds() > 2:
                    save_to_mongodb(track_id, info, camera_number)
                    info['last_save'] = datetime.now()

        # Write frame to output
        out.write(frame)
        cv2.imshow(f'Camera {camera_number} - Vehicle Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return vehicle_info