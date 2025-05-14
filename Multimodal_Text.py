import re
import spacy

# Initialize NLP Model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(query: str):
    """
    Extracts vehicle details, number plate, camera number, and tracking intent from a text query.
    """
    doc = nlp(query.lower())
    extracted_keywords = set()

    # Predefined lists
    vehicle_names = {
        "minivan": ["minivan"],
        "van": ["van"],
        "suv": ["suv", "jeep"],
        "sedan": ["sedan"],
        "truck": ["truck", "pickup"],
        "hatchback": ["hatchback"],
        "motorcycle": ["motorcycle", "bike"]
    }
    colors = {
        "red": ["red", "maroon"],
        "blue": ["blue", "navy"],
        "green": ["green", "emerald"],
        "black": ["black", "onyx"],
        "white": ["white", "ivory"],
        "gray": ["gray", "grey", "silver"],
        "yellow": ["yellow", "gold"],
        "brown": ["brown", "tan"],
        "orange": ["orange", "rust"],
        "purple": ["purple", "violet"]
        }
    company_names = {
        "toyota": ["toyota"],
        "honda": ["honda"],
        "ford": ["ford"],
        "suzuki":["suzuki"],
        "bmw": ["bmw"],
        "audi": ["audi"],
        "mercedes": ["mercedes", "benz"],
        "hyundai": ["hyundai"],
        "tesla": ["tesla"]
    }
    track_keywords = {"track", "follow", "trace", "pursue", "chase", "find path", "locate movement"}

    # Initialize variables
    vehicle_name = None
    color = None
    company = None
    track = False
    description = []

    # Check for tracking intent
    for token in doc:
        if token.text in track_keywords:
            track = True
        elif token.text in vehicle_names:
            vehicle_name = token.text  # Ensure model type detection
        elif token.text in colors:
            color = token.text
        elif token.text in company_names:
            company = token.text.capitalize() 
        elif token.pos_ in {"NOUN", "VERB", "ADJ"}:
            extracted_keywords.add(token.text)
            description.append(token.text)

    # Extract entities
    for ent in doc.ents:
        extracted_keywords.add(ent.text)

    # Enhanced license plate detection (handles more formats)
    plate_patterns = [
        r'\b[A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,2}\s?[0-9]{4}\b',  # KA 01 AB 1234
        r'\b[A-Z]{2}\s?[0-9]{2}\s?[A-Z]{1,2}\s?[0-9]{3,4}\b',   # KA01AB1234
        r'\b[A-Z]{2}\s?[0-9]{4}\s?[A-Z]{1,2}\b',                # KA 1234 AB
        r'\b[A-Z]{3}\s?[0-9]{3,4}\b',                           # ABC 1234
        r'\b[A-Z]{2}\s?[0-9]{3}\s?[A-Z]{1}\b',                  # KA 123 A
        r'\b[0-9]{2}\s?[A-Z]{1,3}\s?[0-9]{4}\b',                # 12 ABC 3456
        r'\b[A-Z]{1,2}\s?[0-9]{3,4}\s?[A-Z]{1,3}\b',            # A 1234 BCD
        r'\b[A-Z]{2}[0-9]{4}[A-Z]{2}\b',                       # KA01AB1234 (no spaces)
    ]
    
    number_plates = []
    for pattern in plate_patterns:
        matches = re.findall(pattern, query.upper())
        number_plates.extend(matches)
    
    number_plate = number_plates[0] if number_plates else None

    # Detect camera number (default to None if not specified)
    camera_match = re.search(r'\b(?:camera|cam|CCTV)\s*(\d+)\b', query, re.IGNORECASE)
    camera_number = int(camera_match.group(1)) if camera_match else None

    # Clean up description
    if vehicle_name:
        description = [word for word in description if word != vehicle_name]
    if color:
        description = [word for word in description if word != color]
    if company:
        description = [word for word in description if word.lower() != company.lower()]

    return {
        "vehicle_name": vehicle_name,
        "color": color,
        "company": company,
        "number_plate": number_plate,
        "camera": camera_number,
        "track": track,
        "description": ' '.join(description)
    }

def get_multimodal_data(query=None):
    if query:
        return extract_keywords(query)
    return None

# if __name__ == "__main__":
#     user_query = input("Enter the vehicle tracking query: ")
#     extracted_data = get_multimodal_data(user_query)

#     print("\nExtracted Information:")
#     for key, value in extracted_data.items():
#         print(f"{key.capitalize()}: {value}")
