#%% load libraries
import json
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ziptie import top_n_bundling

print('* libraries loaded')

#%% load data and create graph
# load data from json file
full_path = os.path.join(
    os.path.dirname(__file__), 
    '../data/migrations.json',  # You can change this to any of your JSON files
)
with open(full_path, 'r') as f:
    data = json.load(f)

print(f'data loaded: {full_path}')
print(f"* loaded {len(data['nodes'])} nodes and {len(data['edges'])} edges")

# create a graph from the data
G = nx.Graph()

# Add nodes with their positions
print(f"* adding {len(data['nodes'])} nodes to the graph ...")
nodes = [(n['id'], { "pos": (n['x'], n['y']) }) for n in data['nodes']]
G.add_nodes_from(nodes)

# Add edges
print(f"* adding {len(data['edges'])} edges to the graph ...")
edges = [(e['source'], e['target']) for e in data['edges']]
G.add_edges_from(edges)

# Extract positions for drawing
print(f"* extracting positions for drawing ...")
pos = nx.get_node_attributes(G, 'pos')

#%% draw the original graph
plt.figure(figsize=(12, 12))
nx.draw(
    G, 
    pos=pos, 
    node_size=30, 
    node_color='skyblue', 
    edge_color='gray',
    width=0.5,
    alpha=0.6
)
plt.title('Original Network Graph')
plt.axis('equal')
plt.savefig('original_graph.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Apply top-N bundling
# Parameters for bundling
n_top_nodes = 15  # Number of top degree nodes to use as control points
radius_factor = 0.15  # Radius of influence relative to graph size
weight_factor = 2.0  # Weight multiplier for control points
distance_threshold = 10  # Minimum edge length to apply bundling

# Apply bundling
bundled_edges, control_points = top_n_bundling.bundle(
    G,
    n_top_nodes=n_top_nodes,
    radius_factor=radius_factor,
    weight_factor=weight_factor,
    distance_threshold=distance_threshold
)

#%% Visualize the bundled network
plt.figure(figsize=(12, 12))

# Draw nodes (small and transparent)
nx.draw_networkx_nodes(
    G, 
    pos=pos, 
    node_size=30, 
    node_color='lightgray', 
    alpha=0.5
)

# Find the min and max weights for color normalization
min_weight = min(cp['weight'] for cp in control_points)
max_weight = max(cp['weight'] for cp in control_points)

# Draw control points with color based on weight
print(f"* drawing control points with color based on weight ...")
for idx, cp in enumerate(control_points):
    # Normalize weight to [0, 1] range for color mapping
    normalized_weight = (cp['weight'] - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5
    
    # Create color: darker red for higher weights
    red_value = 0.7 - normalized_weight * 0.5  # Higher weight = darker red
    color = (red_value, 0, 0)  # Red with varying intensity
    
    # Draw the influence circle with opacity based on weight
    circle_alpha = 0.05 + normalized_weight * 0.15  # Higher weight = more opaque
    circle = plt.Circle((cp['x'], cp['y']), cp['radius'], color=color, alpha=circle_alpha, linewidth=0, edgecolor='none')
    plt.gca().add_patch(circle)
    
    # Draw the control point (hub node)
    plt.scatter(cp['x'], cp['y'], color=color, s=100 + normalized_weight * 200, zorder=10)
    
    # Add label with node ID and degree
    plt.text(cp['x'], cp['y'] + 2, f"Node {cp['node_id']}\nDegree: {cp['degree']}", 
             fontsize=8, ha='center', va='bottom', zorder=10, 
             color='black', bbox=dict(facecolor='white', linewidth=0, alpha=0.7, pad=1))

# Draw bundled edges
print(f"* drawing bundled edges ...")
for edge in tqdm(bundled_edges):
    points = edge['points']
    if len(points) > 2:  # Bezier curve
        xs, ys = zip(*points)
        plt.plot(xs, ys, 'blue', alpha=0.2, linewidth=0.8)
    else:  # Direct line
        plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], 'gray', alpha=0.1, linewidth=0.5)

plt.title(f'Network Graph with Top-{n_top_nodes} Node Bundling')
plt.axis('equal')
plt.savefig('top_n_bundled_graph.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Compare edge counts
total_edges = len(G.edges())
bundled_count = sum(1 for edge in bundled_edges if len(edge['points']) > 2)
direct_count = total_edges - bundled_count

print(f"Total edges: {total_edges}")
print(f"Bundled edges: {bundled_count} ({bundled_count/total_edges*100:.1f}%)")
print(f"Direct edges: {direct_count} ({direct_count/total_edges*100:.1f}%)") 