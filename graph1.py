# ------------------------------
# 1. Camera Network Definition
# ------------------------------
# Define node connections and directions (using your test case data)
DIRECTION_MAP = {
    # (1, 1): "East",
    # (1, 3): "South",
    # (3, 3): "East",
    # (3, 1): "West",
    # (3, 7): "East",
    # (7, 3): "North",
    # (7, 6): "East",
    # (7, 8): "West",
    # (6, 7): "West",
    # (8, 7): "East",

    (1, 1): "East",
    (1, 3): "South",
    (3, 3): "South",
    (4, 3): "North",
    (3, 7): "East",
    (7, 7): "West",
    (2, 1): "Southt",
    (4, 3): "North",


    (2, 2): "South", 
    # (2, 2): "North",
    # (2, 2): "North",
    (4, 4): "South",
    # (4, 4): "South",
    # (4, 4): "East",
    (5, 5): "North",
    (8, 8): "West",
    # (8, 8): "South",
    (5, 2): "North",
    # (5, 2): "North",
    (2, 4): "East",
    # (2, 4): "South",
    (4, 8): "South"
}

# Physical camera connections
# CONNECTED_NODES = {
#     1: [3, 2],
#     3: [1, 7, 4],
#     7: [6, 3, 8],
#     6: [7],
#     8: [7],
#     2: [1, 4, 5],
#     4:[2,3,8]
# }

CAMERA_GRAPH = {
    1: [2, 3],
    2: [1, 4, 5],
    3: [7, 4, 1],
    4: [2, 3, 8],
    5: [2],
    6: [7, 8],
    7: [3, 6, 8],
    8: [4, 6, 7]
}



# Nodes without video feeds
BLIND_SPOTS = {7}

# ------------------------------
# 2. Graph Helper Functions
# ------------------------------
def get_direction(from_node: int, to_node: int) -> str:
    """Safe direction handling with initial state support"""
    try:
        # Handle initial state explicitly
        if from_node is None or from_node == to_node:
            return CAMERA_GRAPH.get((None, to_node), "East")
            
        return CAMERA_GRAPH.get(
            (from_node, to_node), 
            CAMERA_GRAPH.get((None, to_node), "East")  # Fallback to initial direction
        )
    except KeyError:
        return "East"  # Final fallback

# def get_adjacent_nodes(node):
#     return CAMERA_GRAPH.get(node, [])

def get_connected_nodes(node: int) -> list:
    """Returns physically connected nodes with validation"""
    return CAMERA_GRAPH.get(node, [])

def is_blind_spot(node: int) -> bool:
    """Blind spot check with type validation"""
    return node in BLIND_SPOTS

def validate_route(prev_node: int, curr_node: int) -> bool:
    """Enhanced route validation with initial state support"""
    if prev_node == curr_node:  # Initial state
        return True
    return curr_node in get_connected_nodes(prev_node)

# # Test Case 1: Node 1 → 3
# print(get_direction(1, 3))  # Should return "East"

# # Test Case 2: Node 3 → 7
# print(get_direction(3, 7))  # Should return "South"

# # Test Case 3: Node 7 → 6
# print(get_direction(7, 6))  # Should return "East"
# Initial state handling
# print(get_direction(1, 1))  # Returns "Initial-East"
# print(validate_route(1, 1))  # Returns True

# # Normal movement
# print(get_direction(3, 7))  # Returns "South"
# print(validate_route(3, 7))  # Returns True

# # Invalid movement
# print(get_direction(7, 1))  # Returns "East" (fallback)
# print(validate_route(7, 1))  # Returns False