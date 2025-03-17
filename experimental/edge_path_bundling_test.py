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
from ziptie import edge_path_bundling

print('* libraries loaded')


def main():
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

  #%% Apply Edge-Path Bundling
  # Parameters for bundling
  weight_pow = 2.0  # Power to raise edge length for weighting
  k = 2.0  # Maximum distortion threshold
  smoothing = 2  # Smoothing factor for control points
  n_segments = 30  # Number of segments for Bezier curves

  # Apply bundling
  bundled_edges, control_points_info = edge_path_bundling.bundle(
      G,
      weight_pow=weight_pow,
      k=k,
      n_segments=n_segments,
      smoothing=smoothing
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

  # Draw bundled edges
  print(f"* drawing bundled edges...")
  for edge in tqdm(bundled_edges):
      points = edge['points']
      
      if 'is_bundled' in edge and edge['is_bundled']:
          # Bundled edge (curved)
          xs, ys = zip(*points)
          plt.plot(xs, ys, 'blue', alpha=0.3, linewidth=1.0)
      elif 'is_control' in edge and edge['is_control']:
          # Control edge (part of a shortest path)
          plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], 
                  'green', alpha=0.5, linewidth=1.0)
      else:
          # Direct edge (not bundled)
          plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], 
                  'gray', alpha=0.2, linewidth=0.5)

  plt.title(f'Network Graph with Edge-Path Bundling (k={k}, smoothing={smoothing})')
  plt.axis('equal')
  plt.savefig('edge_path_bundled_graph.png', dpi=300, bbox_inches='tight')
  plt.show()

  #%% Compare edge counts
  total_edges = len(G.edges())
  bundled_count = sum(1 for edge in bundled_edges if 'is_bundled' in edge and edge['is_bundled'])
  control_count = sum(1 for edge in bundled_edges if 'is_control' in edge and edge['is_control'])
  direct_count = total_edges - bundled_count - control_count

  print(f"Edge-Path Bundling Statistics:")
  print(f"  Total edges: {total_edges}")
  print(f"  Bundled edges: {bundled_count} ({bundled_count/total_edges*100:.1f}%)")
  print(f"  Control edges: {control_count} ({control_count/total_edges*100:.1f}%)")
  print(f"  Direct edges: {direct_count} ({direct_count/total_edges*100:.1f}%)")

if __name__ == "__main__":
  main()
