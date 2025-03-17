import numpy as np
from scipy.special import comb
from tqdm import tqdm

def point_to_line_distance(point, line_start, line_end):
    '''
    Calculate the distance from a point to a line segment
    
    Parameters:
    -----------
    point: tuple
        The point to calculate the distance to
    line_start: tuple
        The start of the line segment
    line_end: tuple
        The end of the line segment
        
    Returns:
    --------
    distance: float
        The distance from the point to the line segment
    '''
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


def weighted_bezier_curve(points, weights, t):
    """
    Evaluate a weighted Bezier curve at parameter t.
    
    Parameters:
    -----------
    points : list of tuples
        Control points for the Bezier curve
    weights : list of floats
        Weights for each control point (higher = more influence)
    t : float
        Parameter value (0 to 1)
        
    Returns:
    --------
    point : numpy.ndarray
        Point on the Bezier curve at parameter t
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
