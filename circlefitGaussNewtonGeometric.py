import numpy as np
from scipy.optimize import least_squares


def fit_circle_gauss_newton(edge_image, v0=None, max_iter=50, tol=1e-5):
    """
    Fit a circle to edge pixels using nonlinear least squares (geometric distance).

    Parameters:
        edge_image: binary image with edges.
        v0: initial guess (x, y, r) as array-like, or None to auto-compute.
        max_iter: maximum number of iterations.
        tol: tolerance for convergence.

    Returns:
        fitted_circle: dict with xCenter, yCenter, radius.
    """
    # Find edge coordinates
    y, x = np.nonzero(edge_image)

    if len(x) < 3:
        raise ValueError("At least 3 edge pixels are required.")

    coords = np.column_stack((x, y))

    # If no initial guess, compute using basic algebraic circle fit (mean + radius)
    if v0 is None:
        x_m = np.mean(x)
        y_m = np.mean(y)
        r_m = np.mean(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))
        v0 = np.array([x_m, y_m, r_m])

    def residuals(v):
        cx, cy, r = v
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r

    # Run least squares optimization
    res = least_squares(
        residuals,
        v0,
        method='lm',  # Levenberg-Marquardt (Gauss-Newton)
        xtol=tol,
        max_nfev=max_iter
    )

    # Extract result
    fitted_circle = {
        "x_center": res.x[0],
        "y_center": res.x[1],
        "radius": res.x[2]
    }

    return fitted_circle
