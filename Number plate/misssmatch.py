from fuzzywuzzy import fuzz

def match_number_plate(original, detected):
    # Remove unwanted characters (like spaces or underscores) before comparing
    original = original.replace(' ', '').replace('_', '')
    detected = detected.replace(' ', '').replace('_', '')

    # Compare using fuzzy matching ratio
    similarity_ratio = fuzz.ratio(original, detected)
    
    print(f"Similarity Ratio: {similarity_ratio}%")
    
    if similarity_ratio > 60:  # You can set your threshold here
        return True
    else:
        return False

# Test with your plates
original_plate = "TN8BJ9744"
detected_plate = "T881974"
match_result = match_number_plate(original_plate, detected_plate)

print("Match Found:", match_result)
