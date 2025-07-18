import numpy as np
from edge_detection_dependance_function.get_edge_pixel_coordinates import get_edge_pixel_coordinates
from edge_detection_dependance_function.ellipse_get_parameter_from_conic import ellipse_get_parameters_from_conic

def ellipse_fit_least_square_halir(
    edge_image,
    weights=None,
    sigma_noise_squared=-np.inf,
    compute_parametric=True,
    compute_focii_string=True
):
    """
    Fit an ellipse to edge pixels using Halíř and Flusser's method.

    Parameters
    ----------
    edge_image : ndarray
        2D boolean array. True = edge pixels.
    weights : ndarray or None
        1D array of weights per point, or None for uniform weights.
    sigma_noise_squared : float
        If >=0, enables bias correction; if -np.inf disables it.
    compute_parametric : bool
        Whether to compute parametric ellipse parameters.
    compute_focii_string : bool
        Whether to compute focii and string parameters.

    Returns
    -------
    fitted_ellipse : dict
        Dictionary of ellipse parameters.
    lambda_div_delta : float
        Correction term for bias correction.
    """
    # Get edge pixel coordinates
    x, y = get_edge_pixel_coordinates(edge_image)
    n_points = len(x)
    if n_points < 5:
        raise ValueError(f"Need at least 5 pixels, got {n_points}")

    # Weights
    if weights is None:
        w = np.ones(n_points)
    else:
        w = np.asarray(weights)
    sqrt_w = np.sqrt(w)

    # Design matrices
    D1 = np.column_stack((x**2, x*y, y**2)) * sqrt_w[:,None]
    D2 = np.column_stack((x, y, np.ones(n_points))) * sqrt_w[:,None]

    # Scatter matrices
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    # Bias correction
    if sigma_noise_squared >= 0:
        DeltaS1 = np.array([
            [6*S3[0,0],     3*S3[0,1],         S3[0,0]+S3[1,1]],
            [3*S3[0,1],     S3[0,0]+S3[1,1],   3*S3[0,1]],
            [S3[0,0]+S3[1,1],3*S3[0,1],        6*S3[1,1]]
        ])
        DeltaS2 = np.array([
            [3*S3[0,2], S3[1,2],      S3[2,2]],
            [S3[1,2],   S3[0,2],      0],
            [S3[0,2],   3*S3[1,2],    S3[2,2]]
        ])
        DeltaS3 = np.zeros((3,3))
        DeltaS3[0,0] = S3[2,2]
        DeltaS3[1,1] = S3[2,2]

        S1 -= sigma_noise_squared * DeltaS1
        S2 -= sigma_noise_squared * DeltaS2
        S3 -= sigma_noise_squared * DeltaS3

    # Compute T
    T = -np.linalg.inv(S3) @ S2.T

    # Build reduced scatter matrix
    M = S1 + S2 @ T
    M_trans = np.vstack([
        M[2,:]/2,
        -M[1,:],
        M[0,:]/2
    ])

    # Solve eigensystem
    eigvals, eigvecs = np.linalg.eig(M_trans.T)
    # Check ellipse constraint: 4*A*C - B^2 >0
    cond = 4*eigvecs[0,:]*eigvecs[2,:] - eigvecs[1,:]**2
    valid_idx = np.where(cond > 0)[0]
    if len(valid_idx)==0:
        raise ValueError("No valid eigenvector found satisfying ellipse constraint.")
    v = eigvecs[:,valid_idx[0]]

    # Reconstruct conic parameters
    u = np.concatenate([v, T @ v])

    # Bias correction term
    if sigma_noise_squared >=0:
        lambda_ = np.linalg.norm(eigvals[valid_idx[0]])
        DeltaS = np.block([
            [DeltaS1, DeltaS2],
            [DeltaS2.T, DeltaS3]
        ])
        delta = u @ DeltaS @ u
        lambda_div_delta = lambda_ / delta
    else:
        lambda_div_delta = 0.0

    # Compute ellipse parameters
    fitted_ellipse = ellipse_get_parameters_from_conic(
        u,
        compute_parametric=compute_parametric,
        compute_focii_string=compute_focii_string
    )
    return fitted_ellipse, lambda_div_delta
