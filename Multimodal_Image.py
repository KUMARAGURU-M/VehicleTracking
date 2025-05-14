import os
import cv2
import torch
import re
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Set device with CUDA priority
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Class definitions
Modal_classes = ['MPV', 'Minivan']
Vehicle_classes = ['Bus', 'Car', 'Two wheeler']
Number_plate_classes = ['Number Plate']

# Color detection parameters
COLOR_RANGES = {
    'red':    ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
    'blue':   ([100, 150, 50], [140, 255, 255]),
    'dark blue': ([100, 100, 20], [130, 255, 100]),
    'green':  ([40, 50, 50], [80, 255, 255]),
    'white':  ([0, 0, 200], [180, 30, 255]),
    'black':  ([0, 0, 0], [180, 255, 30]),
    'yellow': ([20, 100, 100], [40, 255, 255]),
    'silver': ([0, 0, 150], [180, 30, 210]),
    'gray':   ([0, 0, 50], [180, 30, 150]),
    'orange': ([10, 100, 100], [20, 255, 255]),
}

def detect_vehicle_color(roi):
    """Detect dominant vehicle color using HSV histogram analysis"""
    if roi.size == 0:
        return 'unknown', 0.0
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))

        hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        dominant_hue = np.argmax(hist)

        color_scores = {}
        for color, ranges in COLOR_RANGES.items():
            if len(ranges) == 4:
                lower1, upper1, lower2, upper2 = ranges
                mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                score = (cv2.countNonZero(mask1) + cv2.countNonZero(mask2)) / (roi.size / 3)
            else:
                lower, upper = ranges
                score = cv2.countNonZero(cv2.inRange(hsv, np.array(lower), np.array(upper))) / (roi.size / 3)

            color_scores[color] = score

        best_color = max(color_scores, key=color_scores.get)
        confidence = color_scores[best_color]
        return best_color, confidence
    except:
        return 'unknown', 0.0

# Initialize OCR reader
ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())

# Load models
def load_model(path):
    model = YOLO(path).to(device)
    model.fuse()
    if device == 'cuda':
        model.half()
    return model

# Define model paths
modal_path = r"E:\Unisys3\Modal\blue\best.pt"
numberplate_path = r"E:\Unisys3\Number plate\Latest\best.pt"
vehicle_path = r"E:\Unisys3\Vehicle\Latest\best.pt"

try:
    vehicle_model = load_model(vehicle_path)
    modal_model = load_model(modal_path)
    number_plate_model = load_model(numberplate_path)
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

def extract_camera_and_tracking(query):
    
    # Extract camera number
    camera_match = re.search(r'\b(?:camera|cam)\s*(\d+)\b', query, re.IGNORECASE)
    camera_number = int(camera_match.group(1)) if camera_match else 1
    
    if camera_match:
        query = re.sub(r'\b(?:camera|cam)\s*\d+\b', '', query, flags=re.IGNORECASE).strip()

    # Check for tracking keywords
    tracking_keywords = ["track", "follow", "trace", "pursue", "chase", "find path", "locate movement"]
    track_vehicle = any(keyword in query.lower() for keyword in tracking_keywords)
    
    if track_vehicle:
        for keyword in tracking_keywords:
            query = re.sub(r'\b' + keyword + r'\b', '', query, flags=re.IGNORECASE)
        query = query.strip()

    return camera_number, track_vehicle

@torch.no_grad()
def process_image(image_path, user_query=""):
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    # Extract camera and tracking info from query
    camera_number, track_vehicle= extract_camera_and_tracking(user_query)

    # Initialize output
    output = {
        "vehicle_name": "Unknown",
        "number_plate": "Not detected",
        "modal_type": "Unknown",
        "color": "Unknown",
        "camera": camera_number,
        "track": track_vehicle,
        "description": ""
    }

    # Vehicle detection
    vehicle_results = vehicle_model.predict(frame, conf=0.6, verbose=False, device=device)

    for result in vehicle_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)

            # Update vehicle name
            output["vehicle_name"] = Vehicle_classes[cls]

            # Process vehicle ROI
            vehicle_roi = frame[y1:y2, x1:x2]
            if vehicle_roi.size == 0:
                continue

            # Color detection
            color, color_conf = detect_vehicle_color(vehicle_roi)
            output["color"] = color

            # Modal detection
            try:
                modal_results = modal_model.predict(vehicle_roi, conf=0.5, verbose=False, device=device)
                for modal in modal_results:
                    for mbox, mcls, mconf in zip(modal.boxes.xyxy.cpu().numpy(),
                                                modal.boxes.cls.cpu().numpy().astype(int),
                                                modal.boxes.conf.cpu().numpy()):
                        if mconf > 0.5:  # Confidence threshold
                            output["modal_type"] = Modal_classes[int(mcls)]
            except Exception as e:
                print(f"Modal detection error: {e}")

            # Number plate detection
            try:
                plate_results = number_plate_model.predict(vehicle_roi, conf=0.4, verbose=False, device=device)
                for plate in plate_results:
                    for pbox, pcls, pconf in zip(plate.boxes.xyxy.cpu().numpy(),
                                                plate.boxes.cls.cpu().numpy().astype(int),
                                                plate.boxes.conf.cpu().numpy()):
                        if pconf > 0.4:  # Confidence threshold
                            px1, py1, px2, py2 = map(int, pbox)
                            plate_roi = vehicle_roi[py1:py2, px1:px2]
                            if plate_roi.size > 0:
                                gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                                result = ocr_reader.ocr(gray_plate, cls=True)
                                if result and len(result[0]) > 0:
                                    output["number_plate"] = result[0][0][1][0]
            except Exception as e:
                print(f"Plate processing error: {e}")

    # Save output image
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, frame)
    return output
    # print(output) 

def get_multimodal_data(image_path=None, user_query=""):
    if image_path:
        return process_image(image_path, user_query)
    return None

# if __name__ == "__main__":
#     # Example usage
#     image_path = input("Enter image path: ").strip()
#     user_query = input("Enter your query (optional): ").strip()
#     get_multimodal_data(image_path, user_query)
     