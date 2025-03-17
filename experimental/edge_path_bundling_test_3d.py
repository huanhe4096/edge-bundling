#%% load libraries
import json
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import sys
import random

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

    # Add nodes with their positions (including random z-coordinate)
    print(f"* adding {len(data['nodes'])} nodes to the graph with 3D positions...")
    nodes = []
    for n in data['nodes']:
        # Generate random z-coordinate between -20 and 20
        z = random.uniform(-20, 20)
        nodes.append((n['id'], {"pos": (n['x'], n['y'], z)}))
    
    G.add_nodes_from(nodes)

    # Add edges
    print(f"* adding {len(data['edges'])} edges to the graph ...")
    edges = [(e['source'], e['target']) for e in data['edges']]
    G.add_edges_from(edges)

    # Extract positions for drawing
    print(f"* extracting positions for drawing ...")
    pos = nx.get_node_attributes(G, 'pos')

    #%% draw the original 3D graph
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw nodes
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]
    ax.scatter(xs, ys, zs, s=30, c='skyblue', alpha=0.6)
    
    # Draw edges
    for edge in G.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, 'gray', alpha=0.3, linewidth=0.5)
    
    ax.set_title('Original 3D Network Graph')
    plt.savefig('original_3d_graph.png', dpi=300, bbox_inches='tight')
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

    #%% Visualize the bundled 3D network
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Draw nodes (small and transparent)
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]
    ax.scatter(xs, ys, zs, s=30, c='lightgray', alpha=0.3)

    # Draw bundled edges
    print(f"* drawing bundled edges in 3D...")
    for edge in tqdm(bundled_edges):
        points = edge['points']
        
        if 'is_bundled' in edge and edge['is_bundled']:
            # Bundled edge (curved)
            xs, ys, zs = zip(*points)
            ax.plot(xs, ys, zs, 'blue', alpha=0.3, linewidth=1.0)
        elif 'is_control' in edge and edge['is_control']:
            # Control edge (part of a shortest path)
            xs = [points[0][0], points[1][0]]
            ys = [points[0][1], points[1][1]]
            zs = [points[0][2], points[1][2]]
            ax.plot(xs, ys, zs, 'green', alpha=0.5, linewidth=1.0)
        else:
            # Direct edge (not bundled)
            xs = [points[0][0], points[1][0]]
            ys = [points[0][1], points[1][1]]
            zs = [points[0][2], points[1][2]]
            ax.plot(xs, ys, zs, 'gray', alpha=0.2, linewidth=0.5)

    # Optionally, visualize control points
    if control_points_info:
        cp_xs = [cp['x'] for cp in control_points_info if cp['is_original']]
        cp_ys = [cp['y'] for cp in control_points_info if cp['is_original']]
        cp_zs = [cp['z'] for cp in control_points_info if cp['is_original']]
        ax.scatter(cp_xs, cp_ys, cp_zs, c='red', s=20, alpha=0.7)

    ax.set_title(f'3D Network Graph with Edge-Path Bundling (k={k}, smoothing={smoothing})')
    plt.savefig('edge_path_bundled_3d_graph.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create a few different views of the 3D graph
    for elev, azim in [(30, 45), (0, 0), (0, 90), (90, 0)]:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set the view
        ax.view_init(elev=elev, azim=azim)
        
        # Draw nodes
        ax.scatter(xs, ys, zs, s=30, c='lightgray', alpha=0.3)
        
        # Draw bundled edges
        for edge in bundled_edges:
            points = edge['points']
            
            if 'is_bundled' in edge and edge['is_bundled']:
                # Bundled edge (curved)
                xs, ys, zs = zip(*points)
                ax.plot(xs, ys, zs, 'blue', alpha=0.3, linewidth=1.0)
            elif 'is_control' in edge and edge['is_control']:
                # Control edge (part of a shortest path)
                xs = [points[0][0], points[1][0]]
                ys = [points[0][1], points[1][1]]
                zs = [points[0][2], points[1][2]]
                ax.plot(xs, ys, zs, 'green', alpha=0.5, linewidth=1.0)
        
        ax.set_title(f'3D Bundled Graph View (elev={elev}, azim={azim})')
        plt.savefig(f'bundled_3d_view_elev{elev}_azim{azim}.png', dpi=300, bbox_inches='tight')
        plt.show()

    #%% Compare edge counts
    total_edges = len(G.edges())
    bundled_count = sum(1 for edge in bundled_edges if 'is_bundled' in edge and edge['is_bundled'])
    control_count = sum(1 for edge in bundled_edges if 'is_control' in edge and edge['is_control'])
    direct_count = total_edges - bundled_count - control_count

    print(f"3D Edge-Path Bundling Statistics:")
    print(f"  Total edges: {total_edges}")
    print(f"  Bundled edges: {bundled_count} ({bundled_count/total_edges*100:.1f}%)")
    print(f"  Control edges: {control_count} ({control_count/total_edges*100:.1f}%)")
    print(f"  Direct edges: {direct_count} ({direct_count/total_edges*100:.1f}%)")
    
    # Verify 3D functionality
    print(f"\nVerifying 3D functionality:")
    sample_edge = next(edge for edge in bundled_edges if 'is_bundled' in edge and edge['is_bundled'])
    sample_point = sample_edge['points'][0]
    print(f"  Sample bundled edge point: {sample_point}")
    print(f"  Point dimensionality: {len(sample_point)}")
    print(f"  Is 3D according to metadata: {sample_edge['is_3d']}")

if __name__ == "__main__":
    main() 