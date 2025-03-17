import networkx as nx
import numpy as np
from scipy.special import comb
from tqdm import tqdm

def bundle(
    G,
    n_top_nodes=10,
    radius_factor=0.1,
    weight_factor=1.0,
    n_segments=20,
    distance_threshold=10
):
    """
    Bundle edges in a graph based on the top N nodes with highest degree.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to bundle edges for
    n_top_nodes : int, optional
        Number of top degree nodes to use as control points (default: 10)
    radius_factor : float, optional
        Factor to determine the radius of influence for each control point
        relative to the graph size (default: 0.1)
    weight_factor : float, optional
        Factor to determine the weight of each control point based on its degree
        (default: 1.0)
    n_segments : int, optional
        Number of segments to use for Bezier curves (default: 20)
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
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Calculate graph size for radius scaling
    x_values = [p[0] for p in pos.values()]
    y_values = [p[1] for p in pos.values()]
    graph_width = max(x_values) - min(x_values)
    graph_height = max(y_values) - min(y_values)
    graph_size = max(graph_width, graph_height)
    
    # Find top N nodes with highest degree
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:n_top_nodes]
    
    print(f"Using top {n_top_nodes} nodes as control points:")
    for i, (node, degree) in enumerate(top_nodes):
        print(f"  {i+1}. Node {node}: degree {degree}")
    
    # Create control points from top nodes
    control_points = []
    for node, degree in top_nodes:
        if node in pos:
            # Scale radius based on graph size and degree
            radius = radius_factor * graph_size
            # Scale weight based on degree
            weight = degree * weight_factor
            
            control_points.append({
                'x': pos[node][0],
                'y': pos[node][1],
                'weight': weight,
                'radius': radius,
                'node_id': node,
                'degree': degree
            })
    
    # Now apply bundling using these control points
    bundled_edges = []
    
    # Function to calculate distance from a point to a line segment
    def point_to_line_distance(point, line_start, line_end):
        # Convert to numpy arrays for vector operations
        p = np.array(point)
        a = np.array(line_start)
        b = np.array(line_end)
        
        # Vector from a to b
        ab = b - a
        # Vector from a to p
        ap = p - a
        
        # Calculate the projection of ap onto ab
        ab_norm = np.linalg.norm(ab)
        
        # Handle case where line is just a point
        if ab_norm < 1e-10:
            return np.linalg.norm(ap)
        
        # Normalized ab vector
        ab_unit = ab / ab_norm
        
        # Projection of ap onto ab_unit
        projection = np.dot(ap, ab_unit)
        
        # Clamp projection to [0, |ab|]
        projection = max(0, min(ab_norm, projection))
        
        # Calculate the closest point on the line segment
        closest_point = a + projection * ab_unit
        
        # Return the distance from p to the closest point
        return np.linalg.norm(p - closest_point)
    
    # Function to evaluate a weighted Bezier curve at parameter t
    def weighted_bezier_curve(points, weights, t):
        """
        Evaluate a weighted Bezier curve at parameter t.
        """
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]
        
        # For weighted Bezier, we'll use a modified approach:
        # 1. First calculate regular Bezier basis functions
        n = len(points) - 1
        basis = [comb(n, i) * (1 - t) ** (n - i) * t ** i for i in range(n + 1)]
        
        # 2. Apply weights to the basis functions and renormalize
        weighted_basis = [b * w for b, w in zip(basis, norm_weights)]
        basis_sum = sum(weighted_basis)
        if basis_sum > 0:  # Avoid division by zero
            weighted_basis = [b / basis_sum for b in weighted_basis]
        
        # 3. Calculate the weighted point
        point = np.zeros(2)
        for i, p in enumerate(points):
            point += weighted_basis[i] * np.array(p)
            
        return point
    
    # Process each edge
    bundled_edges_count = 0
    print(f"Processing {len(G.edges())} edges...")
    for u, v in tqdm(G.edges()):
        source_pos = pos[u]
        target_pos = pos[v]
        
        # Calculate direct distance between nodes
        direct_distance = np.linalg.norm(np.array(source_pos) - np.array(target_pos))
        
        # If the edge is too short, just use a direct line
        if direct_distance < distance_threshold:
            bundled_edges.append({
                'source': u,
                'target': v,
                'points': [source_pos, target_pos]
            })
            continue
        
        # Skip self-loops
        if u == v:
            continue
            
        # Skip edges connected to control points (optional - can be commented out)
        # if u in [cp['node_id'] for cp in control_points] or v in [cp['node_id'] for cp in control_points]:
        #     bundled_edges.append({
        #         'source': u,
        #         'target': v,
        #         'points': [source_pos, target_pos]
        #     })
        #     continue
        
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
        else:
            # Generate points along the weighted Bezier curve
            curve_points = [weighted_bezier_curve(bezier_controls, bezier_weights, t) 
                           for t in np.linspace(0, 1, n_segments)]
            bundled_edges.append({
                'source': u,
                'target': v,
                'points': curve_points
            })
            bundled_edges_count += 1

    print(f"Number of edges bundled: {bundled_edges_count}")
    return bundled_edges, control_points