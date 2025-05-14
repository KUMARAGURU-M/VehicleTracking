import networkx as nx
import matplotlib.pyplot as plt

# Define the updated directions with A → C as North
G = nx.DiGraph()

directions = {
    'A': {'B': 'East', 'C': 'North'},  # A to C is now North
    'B': {'A': 'West', 'D': 'North', 'E': 'East'},
    'C': {'A': 'South', 'D': 'East'},
    'D': {'B': 'South', 'C': 'West', 'G': 'North'},
    'E': {'B': 'West'},
    'G': {'D': 'South', 'H': 'East'},
    'H': {'G': 'West'}
}

# Add edges to the graph
for node, neighbors in directions.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor, direction=neighbors[neighbor])

# Define positions for visualization (same layout)
positions = {
    'A': (0, 0), 'B': (2, 0), 'C': (0, 2), 'D': (2, 2),
    'E': (4, 0), 'G': (2, 4), 'H': (4, 4)
}

# Draw the graph
plt.figure(figsize=(6, 6))
nx.draw(G, positions, with_labels=True, node_size=700, node_color="lightblue", edge_color="black", font_size=10, font_weight="bold", arrows=True)

# Add edge labels for directions
edge_labels = {(node, neighbor): directions[node][neighbor] for node in directions for neighbor in directions[node]}
nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, font_size=8)

# Show the plot
plt.title("Directional Graph with Corrected A → C Direction")
plt.show()
