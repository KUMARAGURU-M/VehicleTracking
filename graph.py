import networkx as nx

# Define the graph with 5 nodes and the given connections
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5 ,6, 7, 8])
G.add_edges_from([
    (1, 2), (1, 3),
    (2, 4), (2, 5),
    (3, 4), (3, 7),
    (4, 8),
    (7, 6), (7, 8),
    (6, 8) 
])

# Tracked path
tracked_path = []

def check_camera(camera):
    return camera if camera in G.nodes else None

def get_connected_nodes(camera):
    """Return connected nodes while maintaining the tracking list."""
    global tracked_path
    if camera not in G.nodes:
        return []

    neighbors = list(G.neighbors(camera))
    tracked_path.append(camera)  # Store visited camera

    # Move previous node to the end of the list for logical traversal
    if len(tracked_path) > 1:
        prev_node = tracked_path[-2]  # Second last visited node
        if prev_node in neighbors:
            neighbors.remove(prev_node)
            neighbors.append(prev_node)
    
    return neighbors
              
                 
            