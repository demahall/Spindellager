from scipy.interpolate import splprep, splev
import numpy as np

def interparc(N, x, y):
    # parametrize by arc length
    pts = np.column_stack((x, y))
    dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    u = np.insert(np.cumsum(dists), 0, 0)
    u /= u[-1]

    # Fit spline
    tck, _ = splprep([x, y], u=u, s=0)

    # New parameter values
    u_fine = np.linspace(0, 1, N)

    # Evaluate spline
    x_new, y_new = splev(u_fine, tck)

    return np.column_stack((x_new, y_new))
