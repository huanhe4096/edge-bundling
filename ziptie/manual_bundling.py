import networkx as nx
import numpy as np
from .utils import point_to_line_distance, weighted_bezier_curve
from tqdm import tqdm


def bundle(
    G, 
    control_points, 
    n_segments=50,
    distance_threshold=10
):
    """
    Bundle edges in a graph based on control points.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to bundle edges for
    control_points : list of dict
        List of control points. Each control point is a dict with keys:
        - x: x-coordinate
        - y: y-coordinate
        - weight: influence weight of the control point (higher = more influence)
        - radius: radius of influence
    distance_threshold : float, optional
        Minimum edge length to apply bundling (default: 10)
        
    Returns:
    --------
    bundled_edges : list of dict
        List of bundled edges. Each edge is a dict with keys:
        - source: source node ID
        - target: target node ID
        - points: list of (x, y) coordinates defining the bezier curve
    """
    pos = nx.get_node_attributes(G, 'pos')
    bundled_edges = []
    

    # Process each edge
    # count the number of edges bundled
    bundled_edges_count = 0
    unbundled_edges_count = 0
    for u, v in tqdm(G.edges(), desc="Bundling edges", total=len(G.edges())):
        source_pos = pos[u]
        target_pos = pos[v]
        
        # Calculate direct distance between nodes
        direct_distance = np.linalg.norm(np.array(source_pos) - np.array(target_pos))
        
        # If the edge is too short, just use a direct line
        if direct_distance <= distance_threshold:
            bundled_edges.append({
                'source': u,
                'target': v,
                'points': [source_pos, target_pos]
            })
            unbundled_edges_count += 1
            continue
        
        # Start with source and target as control points and weights
        bezier_controls = [source_pos]
        bezier_weights = [1.0]  # Default weight for source
        
        # Check each control point
        for cp in control_points:
            cp_pos = (cp['x'], cp['y'])
            distance = point_to_line_distance(cp_pos, source_pos, target_pos)
            
            # If the edge passes within the radius of the control point, add it
            if distance <= cp['radius']:
                # Add the control point with its weight
                bezier_controls.append(cp_pos)
                
                # Calculate influence based on distance and weight
                # The closer to the line, the more influence it has
                influence = cp['weight'] * (1 - distance / cp['radius'])
                bezier_weights.append(influence)
        
        # Add target as the final control point
        bezier_controls.append(target_pos)
        bezier_weights.append(1.0)  # Default weight for target
        
        # If no control points were added, just use a direct line
        if len(bezier_controls) == 2:
            bundled_edges.append({
                'source': u,
                'target': v,
                'points': [source_pos, target_pos]
            })
            unbundled_edges_count += 1
        else:
            # Generate points along the weighted Bezier curve
            curve_points = [
                weighted_bezier_curve(bezier_controls, bezier_weights, t) 
                for t in np.linspace(0, 1, n_segments)
            ]
            bundled_edges.append({
                'source': u,
                'target': v,
                'points': curve_points
            })
            bundled_edges_count += 1

    print(f"Number of edges bundled: {bundled_edges_count}")
    print(f"Number of edges unbundled: {unbundled_edges_count}")

    return bundled_edges