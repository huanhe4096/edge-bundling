#%% load libraries
import json
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from tqdm import tqdm
print('* libraries loaded')


#%% create a graph of small world network
# load data from json file
full_path = os.path.join(
    os.path.dirname(__file__), 
    # '../data/2n1e.json',
    # '../data/migrations.json',
    '../data/glp1.json',
)
with open(full_path, 'r') as f:
    data = json.load(f)

print(f'data loaded: {full_path}')
print(f"* loaded {len(data['nodes'])} nodes and {len(data['edges'])} edges")

# data contains {nodes: [{x, y, id}], edges: [{source: ..., target: ...,}]}
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

# draw the graph with the original positions
plt.figure(figsize=(10, 10))
nx.draw(
    G, 
    pos=pos, 
    # with_labels=True, 
    node_size=100, 
    node_color='skyblue', 
    edge_color='gray'
  )

plt.title('Network Graph with Original Positions')
plt.axis('equal')
plt.show()


#%% Define the edge bundling function


#%% Example of using the bundle function
# random generate 100 control points
pos = nx.get_node_attributes(G, 'pos')
max_x = max(p[0] for p in pos.values())
max_y = max(p[1] for p in pos.values())

# Define some control points
control_points = [
    # {'x': -50, 'y': 20, 'weight': 10, 'radius': 30},
    {'x': -50, 'y': 20, 'weight': 5, 'radius': 30},
    {'x': 50, 'y':  40, 'weight': 1, 'radius': 55},
    # {'x': -98, 'y': 2, 'weight': 1, 'radius': 10},
    # # {'x': -20, 'y': -4, 'weight': 1, 'radius': 10},
    # {'x': -25, 'y': 12, 'weight': 100, 'radius': 15},
    # {'x': 72, 'y': 12, 'weight': 1, 'radius': 10},

    # {'x': 60, 'y': 6, 'weight': 2, 'radius': 15},
    # {'x': 55, 'y': -6, 'weight': 10, 'radius': 25},
    # {'x': 40, 'y': -22, 'weight': 100, 'radius': 10},
    # {'x': 10, 'y': -30, 'weight': 2, 'radius': 20},
    # {'x': -90, 'y': 10, 'weight': 2, 'radius': 10},

    # {'x': -50, 'y': -12, 'weight': 1, 'radius': 10},

]

# control_points = [
#     {'x': np.random.randint(-max_x, max_x), 'y': np.random.randint(-max_y, max_y), 'weight': np.random.randint(1, 10), 'radius': 5}
#     for _ in range(20)
# ]

# just generate 20 control points at the center line of the graph from left to right
# control_points = [
#     {'x': i, 'y': 0, 'weight': 10, 'radius': 30}
#     for i in range(int(-max_x), int(max_x))
# ]

# Apply bundling
# load the manual bundling function
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ziptie import manual_bundling
# reload this file
from importlib import reload
reload(manual_bundling)

bundled_edges = manual_bundling.bundle(
    G, 
    control_points,
    distance_threshold=10
)

# Visualize the bundled network
plt.figure(figsize=(10, 10))

# Draw nodes
nx.draw_networkx_nodes(G, pos=pos, node_size=100, node_color='skyblue')
# nx.draw_networkx_labels(G, pos=pos, font_size=6)

# Find the min and max weights for color normalization
min_weight = min(cp['weight'] for cp in control_points)
max_weight = max(cp['weight'] for cp in control_points)

# Draw control points with color based on weight
print(f"* drawing control points with color based on weight ...")
for idx, cp in enumerate(control_points):
    # Normalize weight to [0, 1] range for color mapping
    normalized_weight = (cp['weight'] - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5
    
    # Create color: darker red for higher weights
    # RGB values for red channel from 0.7 (light) to 0.2 (dark)
    red_value = 0.7 - normalized_weight * 0.5  # Higher weight = darker red
    color = (red_value, 0, 0)  # Red with varying intensity
    
    # Draw the influence circle with opacity based on weight
    circle_alpha = 0.1 + normalized_weight * 0.3  # Higher weight = more opaque
    circle = plt.Circle((cp['x'], cp['y']), cp['radius'], color=color, alpha=circle_alpha, linewidth=0, edgecolor='none')
    plt.gca().add_patch(circle)
    
    # Draw the control point
    plt.scatter(cp['x'], cp['y'], color=color, s=50 + normalized_weight * 100, zorder=10)
    
    # Add label with weight information
    plt.text(cp['x'], cp['y'] + 2, f"{idx}", 
             fontsize=6, ha='center', va='bottom', zorder=10, 
             color='black', bbox=dict(facecolor='white', linewidth=0, alpha=0.7, pad=1))
    

print(f"* drawing bundled edges ...")
# Draw bundled edges
for edge in tqdm(bundled_edges):
    points = edge['points']
    if len(points) > 2:  # Bezier curve
        xs, ys = zip(*points)
        plt.plot(xs, ys, 'blue', alpha=0.1, linewidth=1)

    else:  # Direct line
        plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], 'gray', alpha=0.1, linewidth=1)

plt.title('Network Graph with Edge Bundling (Control Point Color by Weight)')
plt.axis('equal')
plt.show()



#%% find the shortest path between two nodes
# Get a list of node IDs to ensure they exist
node_ids = list(G.nodes())

# Choose source and target nodes that exist in the graph
source_node = node_ids[0]  # First node
target_node = node_ids[100]  # 100th node

# find the shortest path between two nodes
shortest_path_nodes = nx.shortest_path(G, source=source_node, target=target_node)
print(f"Shortest path from {source_node} to {target_node}: {shortest_path_nodes}")

plt.figure(figsize=(8, 8))

# draw the full graph
nx.draw(G, pos=pos, with_labels=True, font_size=label_size, node_size=node_size, 
        node_color=node_color, edge_color=edge_color, alpha=0.3)

# Create edges for the shortest path
shortest_path_edges = [(shortest_path_nodes[i], shortest_path_nodes[i+1]) for i in range(len(shortest_path_nodes)-1)]

# Highlight the shortest path
nx.draw_networkx_nodes(G, pos=pos, nodelist=shortest_path_nodes, node_color='red', node_size=node_size)
nx.draw_networkx_edges(G, pos=pos, edgelist=shortest_path_edges, edge_color='red', width=2)

plt.title('Network Graph with Shortest Path Highlighted')
plt.show()



