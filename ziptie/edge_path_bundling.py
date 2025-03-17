import networkx as nx
import math
import numpy as np
from tqdm import tqdm
from scipy.special import comb


def bundle(
    G,
    weight_pow=2.0,
    k=2.0,
    n_segments=30,
    smoothing=2,
    generate_curve_points=True
):
    """
    Apply Edge-Path Bundling to a graph with 2D or 3D node positions.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to bundle edges for. Node positions should be stored in the 'pos' attribute
        as tuples or arrays of length 2 (2D) or 3 (3D).
    weight_pow : float, optional
        Power to raise edge length to for weighting (default: 2.0)
    k : float, optional
        Maximum distortion threshold (default: 2.0)
    n_segments : int, optional
        Number of segments for Bezier curves (default: 30)
    smoothing : int, optional
        Smoothing factor for control points (default: 2)
    generate_curve_points : bool, optional
        Whether to generate curve points for visualization (default: True)
        If False, only control points will be included and 'points' will be empty
        
    Returns:
    --------
    bundled_edges : list of dict
        List of bundled edges. Each edge is a dict with keys:
        - source: source node ID
        - target: target node ID
        - points: list of coordinates defining the bezier curve (empty if generate_curve_points=False)
        - control_points: list of coordinates of the control points
    control_points_info : list of dict
        Information about control points used for visualization
    """
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Determine if we're working with 2D or 3D points
    sample_pos = next(iter(pos.values()))
    is_3d = len(sample_pos) == 3
    
    # Create a working copy of the graph with weighted edges
    nx_graph = G.copy()
    
    # Calculate edge weights based on distance
    for u, v in nx_graph.edges():
        u_pos = np.array(pos[u])
        v_pos = np.array(pos[v])
        distance = np.linalg.norm(v_pos - u_pos)
        weight = pow(distance, weight_pow)
        nx_graph[u][v]['distance'] = distance
        nx_graph[u][v]['weight'] = weight
    
    # Sort edges by weight in descending order
    edges = list(nx_graph.edges())
    edge_weights = [nx_graph[u][v]['weight'] for u, v in edges]
    sorted_edges = [edge for _, edge in sorted(zip(edge_weights, edges), reverse=True)]
    
    # Track locked edges (edges that are part of a path)
    locked_edges = set()
    
    # Track edges to skip from shortest path calculations
    skip_edges = set()
    
    # Get all control points
    control_point_lists = []
    bundled_edges_map = {}  # Map edge to its control points
    too_long = 0
    no_path = 0
    
    # Process edges in decreasing weight order
    print(f"Processing {len(sorted_edges)} edges for bundling...")
    for edge in tqdm(sorted_edges, desc="Computing edge paths"):
        u, v = edge
        
        # Skip locked edges
        if edge in locked_edges or (v, u) in locked_edges:
            continue
        
        # Mark edge to skip from shortest path calculations
        skip_edges.add(edge)
        skip_edges.add((v, u))  # Add reverse edge too
        
        # Create a temporary graph without skipped edges
        temp_graph = nx_graph.copy()
        for skip_edge in skip_edges:
            if temp_graph.has_edge(*skip_edge):
                temp_graph.remove_edge(*skip_edge)
        
        # Find shortest path between endpoints using NetworkX
        try:
            if not nx.has_path(temp_graph, u, v):
                no_path += 1
                skip_edges.remove(edge)
                skip_edges.remove((v, u))
                continue
                
            # Get the shortest path nodes
            path_nodes = nx.shortest_path(temp_graph, u, v, weight='weight')
            
            # Get the edges in the path
            path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
            
            # Calculate original edge distance and new path length
            u_pos = np.array(pos[u])
            v_pos = np.array(pos[v])
            original_distance = np.linalg.norm(v_pos - u_pos)
            
            # Calculate path length
            path_length = 0
            for path_u, path_v in path_edges:
                path_u_pos = np.array(pos[path_u])
                path_v_pos = np.array(pos[path_v])
                path_length += np.linalg.norm(path_v_pos - path_u_pos)
            
            # If path is too long (exceeds distortion threshold), don't bundle
            if path_length > k * original_distance:
                too_long += 1
                skip_edges.remove(edge)
                skip_edges.remove((v, u))
                continue
            
            # Lock all edges in the path
            for path_edge in path_edges:
                locked_edges.add(path_edge)
                locked_edges.add((path_edge[1], path_edge[0]))  # Add reverse edge too
            
            # Get control points for drawing
            control_points = get_control_points_from_path(path_nodes, pos, smoothing)
            control_point_lists.append(control_points)
            bundled_edges_map[edge] = control_points
            
        except nx.NetworkXNoPath:
            no_path += 1
            skip_edges.remove(edge)
            skip_edges.remove((v, u))
            continue
    
    print(f"Bundling results:")
    print(f"  Edges processed: {len(sorted_edges)}")
    print(f"  Edges bundled: {len(control_point_lists)}")
    print(f"  Edges not bundled (no path): {no_path}")
    print(f"  Edges not bundled (too long): {too_long}")
    
    # Create bundled edges
    bundled_edges = []
    control_points_info = []
    
    # Process original edges
    for u, v in G.edges():
        edge = (u, v)
        reverse_edge = (v, u)
        
        # Check if this edge is bundled
        if edge in bundled_edges_map:
            # Get control points
            control_points = bundled_edges_map[edge]
            # Convert control points to list of tuples for output
            control_points_list = [tuple(cp) for cp in control_points]
            
            # Create Bezier curve if requested
            curve_points = []
            if generate_curve_points:
                curve_points = create_bezier_curve(control_points, n_segments)
                
            bundled_edges.append({
                'source': u,
                'target': v,
                'points': curve_points,
                'control_points': control_points_list,
                'is_bundled': True,
                'is_3d': is_3d
            })
        elif reverse_edge in bundled_edges_map:
            # Get control points
            control_points = bundled_edges_map[reverse_edge]
            # Convert control points to list of tuples for output
            control_points_list = [tuple(cp) for cp in control_points]
            
            # Create Bezier curve if requested
            curve_points = []
            if generate_curve_points:
                curve_points = create_bezier_curve(control_points, n_segments)
                
            bundled_edges.append({
                'source': u,
                'target': v,
                'points': curve_points,
                'control_points': control_points_list,
                'is_bundled': True,
                'is_3d': is_3d
            })
        else:
            # This is a direct edge or a control edge
            is_control = edge in locked_edges or reverse_edge in locked_edges
            # For direct edges, control points are just the endpoints
            control_points_list = [pos[u], pos[v]]
            
            # Create points list if requested
            curve_points = []
            if generate_curve_points:
                curve_points = [pos[u], pos[v]]
                
            bundled_edges.append({
                'source': u,
                'target': v,
                'points': curve_points,
                'control_points': control_points_list,
                'is_bundled': False,
                'is_control': is_control,
                'is_3d': is_3d
            })
    
    # Create control points info for visualization
    for i, cps in enumerate(control_point_lists):
        for j, cp in enumerate(cps):
            if j > 0 and j < len(cps) - 1:  # Skip source and target
                cp_info = {
                    'is_original': j % smoothing == 0,  # True if this is an original node, not an inserted one
                    'bundle_id': i,
                    'is_3d': is_3d
                }
                
                # Add coordinates based on dimensionality
                if is_3d:
                    cp_info.update({'x': cp[0], 'y': cp[1], 'z': cp[2]})
                else:
                    cp_info.update({'x': cp[0], 'y': cp[1]})
                    
                control_points_info.append(cp_info)
    
    return bundled_edges, control_points_info


def get_control_points_from_path(path_nodes, pos, smoothing):
    """Extract control points from a path using node positions"""
    control_points = []
    
    # Add all nodes in the path as control points
    for node in path_nodes:
        control_points.append(np.array(pos[node]))
    
    # Apply smoothing
    return split_control_points(control_points, smoothing)


def split_control_points(points, smoothing):
    """Apply smoothing by inserting additional control points"""
    p = points
    # For each level of smoothing, insert new control point in the middle of all points
    for smooth in range(1, smoothing):
        new_points = []
        for i in range(len(p) - 1):
            new_point = np.divide(p[i] + p[i + 1], 2.0)
            new_points.append(p[i])
            new_points.append(new_point)
        new_points.append(p[-1])
        p = new_points
    return p


def create_bezier_curve(control_points, n_segments):
    """Create a Bezier curve from control points"""
    curve_points = []
    
    # Function to evaluate a Bezier curve at parameter t
    def bezier_point(t):
        n = len(control_points) - 1
        point = np.zeros(control_points[0].shape)  # Create point with same dimensions as control points
        for i, p in enumerate(control_points):
            point += comb(n, i) * (1 - t) ** (n - i) * t ** i * p
        return point
    
    # Generate points along the curve
    for t in np.linspace(0, 1, n_segments):
        point = bezier_point(t)
        curve_points.append(tuple(point))
    
    return curve_points