from scipy.spatial import cKDTree
import numpy as np

def sort_pixels_according_to_relative_distance(points, max_dist=30):
    """
    Greedy nearest-neighbor sorting of pixels.

    Parameters:
        points (ndarray): shape (N,2)
        max_dist (float): maximum allowed distance between consecutive points

    Returns:
        ndarray: sorted points (N,2)
    """
    points = np.asarray(points)
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    sorted_pts = []

    # Start with the first point
    idx = 0
    sorted_pts.append(points[idx])
    visited[idx] = True

    tree = cKDTree(points)

    for _ in range(n - 1):
        dists, inds = tree.query(points[idx], k=n)
        for j in range(1, len(inds)):
            if not visited[inds[j]] and dists[j] <= max_dist:
                idx = inds[j]
                visited[idx] = True
                sorted_pts.append(points[idx])
                break
        else:
            # No more reachable points
            break
    return np.array(sorted_pts)
