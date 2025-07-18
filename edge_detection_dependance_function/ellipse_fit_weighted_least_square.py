import numpy as np
from edge_detection_dependance_function.get_edge_pixel_coordinates import get_edge_pixel_coordinates
from edge_detection_dependance_function.ellipse_fit_least_square_halir import ellipse_fit_least_square_halir
from edge_detection_dependance_function.ellipse_get_parameter_from_conic import ellipse_get_parameters_from_conic
from edge_detection_dependance_function.compute_rowwise_norm import compute_rowwise_norm
from edge_detection_dependance_function.rotate_vector_by_angle_and_shift import rotate_vector_by_angle_and_shift


def ellipse_fit_weighted_least_square(
    edge_image,
    u0=None,
    w0=None,
    tol_weights=1e-5,
    tol_sigma_noise_squared=1e-5,
    max_iter=50,
    residual_fct='Algebraic',
    sigma_estimate='Barnett',
    weight_fct='Huber',
    iter_non_convex=0,
    compute_parametric=True,
    compute_focii_string=True,
    bias_correction=False
):
    """
    Iterative weighted least squares ellipse fitting.

    Parameters
    ----------
    edge_image : ndarray
        2D binary image.
    u0 : ndarray or None
        Initial conic parameters (6,).
    w0 : ndarray or None
        Initial weights.
    tol_weights : float
        Tolerance for weight convergence.
    tol_sigma_noise_squared : float
        Tolerance for sigma convergence.
    max_iter : int
        Max iterations.
    residual_fct : {'Algebraic','Rosin'}
        Residual function.
    sigma_estimate : {'Barnett','Zhang'}
        Sigma estimation method.
    weight_fct : {'LeastSquare','LeastPower','Fair','Huber','Tukey'}
        Weighting function.
    iter_non_convex : int
        Additional Tukey iterations.
    compute_parametric : bool
        Compute ellipse parameters.
    compute_focii_string : bool
        Compute focii.
    bias_correction : bool
        Enable bias correction.

    Returns
    -------
    fitted_ellipse : dict
        Fitted ellipse parameters.
    sigma_noise : float
        Estimated noise std.
    edge_pixels_and_residuals : ndarray, shape (N,3)
        [x,y,residual] per point.
    """
    # Get coordinates
    x, y = get_edge_pixel_coordinates(edge_image)
    n_points = len(x)
    if n_points < 5:
        raise ValueError("Need at least 5 pixels.")

    # Initialize u0 if None
    if u0 is None:
        ell0, _ = ellipse_fit_least_square_halir(
            edge_image,
            compute_parametric=False,
            compute_focii_string=False
        )
        u0 = ell0["u"]

    u0 = np.asarray(u0).reshape(6)
    # Initial ellipse
    current_ellipse = ellipse_get_parameters_from_conic(u0, True, True)

    iter_count = 0
    weights = np.ones(n_points)
    weights_old = np.full(n_points, np.inf)
    sigma_noise_squared = 0 if bias_correction else -np.inf
    lambda_div_delta = np.inf


    # Start iterations
    while (np.linalg.norm(weights - weights_old) > tol_weights or lambda_div_delta > tol_sigma_noise_squared) and (iter_count < max_iter):
        iter_count += 1
        weights_old = weights.copy()

        # Compute residuals
        if residual_fct == "Algebraic":
            compute_focii_string = False
            A,B,C,D,E,F = current_ellipse["u"]
            residuals = (
                A*x**2 + B*x*y + C*y**2 + D*x + E*y + F
            )
        elif residual_fct == "Rosin":
            compute_focii_string = True
            residuals = _residuals_rosin(current_ellipse, x, y)
        else:
            raise ValueError("Unknown residualFct.")

        # Estimate sigma
        abs_res = np.abs(residuals)
        if sigma_estimate == "Barnett":
            sigma = 1.4826 * np.median(abs_res)
        elif sigma_estimate == "Zhang":
            sigma = 1.4826 * (1 + 5/(len(residuals)-5)) * np.median(abs_res)
        else:
            raise ValueError("Unknown sigmaEstimate.")

        # Compute weights
        if iter_count == 1 and w0 is not None:
            weights = w0
        else:
            if weight_fct == "LeastSquare":
                weights = np.ones_like(residuals)
            elif weight_fct == "LeastPower":
                c = 1.2
                weights = abs_res**(c-2)
            elif weight_fct == "Fair":
                c = 1.3998
                weights = 1/(1 + abs_res/c)
            elif weight_fct == "Huber":
                c = 1.345
                weights = np.where(abs_res <= c*sigma, 1, c*sigma/abs_res)
            elif weight_fct == "Tukey":
                c = 4.6851
                t = abs_res / (c*sigma)
                weights = (1 - t**2)**2
                weights[t > 1] = 0
            else:
                raise ValueError("Unknown weightFct.")

        # Compute new ellipse
        current_ellipse, lambda_div_delta = ellipse_fit_least_square_halir(
            edge_image,
            weights=weights,
            compute_parametric=False,
            compute_focii_string=compute_focii_string,
            sigma_noise_squared=sigma_noise_squared
        )

        sigma_noise_squared = 0 if sigma_noise_squared<0 else sigma_noise_squared + lambda_div_delta

    if iter_count >= max_iter:
        print("Warning: Did not converge in {} iterations.".format(max_iter))

    # Optionally do non-convex Tukey iterations
    if iter_non_convex > 0:
        return ellipse_fit_weighted_least_square(
            edge_image,
            u0=current_ellipse["u"],
            w0=weights,
            tol_weights=-np.inf,
            tol_sigma_noise_squared=-np.inf,
            max_iter=iter_non_convex,
            residual_fct=residual_fct,
            sigma_estimate=sigma_estimate,
            weight_fct='Tukey',
            iter_non_convex=0,
            compute_parametric=compute_parametric,
            compute_focii_string=compute_focii_string,
            bias_correction=bias_correction
        )

    # Final ellipse
    fitted_ellipse, _ = ellipse_fit_least_square_halir(
        edge_image,
        weights=weights,
        compute_parametric=compute_parametric,
        compute_focii_string=compute_focii_string
    )

    sigma_noise = np.sqrt(max(sigma_noise_squared,0))
    edge_pixels_and_residuals = np.column_stack([x,y,residuals])
    return fitted_ellipse, sigma_noise, edge_pixels_and_residuals


def _residuals_rosin(ellipse, x, y):
    """
    Compute Rosin residuals.
    """
    # Compute incenters
    F1 = np.array(ellipse["F1"])
    F2 = np.array(ellipse["F2"])
    opp1 = compute_rowwise_norm(F2 - np.column_stack([x,y]))
    opp2 = compute_rowwise_norm(F1 - np.column_stack([x,y]))
    opp_edge = np.linalg.norm(F1 - F2)

    incenters = (
        (opp1[:,None]*F1 + opp2[:,None]*F2 + opp_edge*np.column_stack([x,y]))
        / (opp1 + opp2 + opp_edge)[:,None]
    )

    yc = -ellipse["yCenter"]
    X_norm = rotate_vector_by_angle_and_shift(np.column_stack([x - ellipse["xCenter"], y - yc]), -ellipse["alpha"])
    I_norm = rotate_vector_by_angle_and_shift(np.column_stack([incenters[:,0] - ellipse["xCenter"], incenters[:,1] - yc]), -ellipse["alpha"])

    # Slope and intercept
    m = (X_norm[:,1]-I_norm[:,1])/(X_norm[:,0]-I_norm[:,0])
    c = X_norm[:,1] - m*X_norm[:,0]

    A = ellipse["b"]**2 + ellipse["a"]**2 * m**2
    B = 2*m*c*ellipse["a"]**2
    C = ellipse["a"]**2*(c**2 - ellipse["b"]**2)

    discrim = np.sqrt(B**2 - 4*A*C)
    ix1 = (-B + discrim)/(2*A)
    ix2 = (-B - discrim)/(2*A)
    iy1 = m*ix1 + c
    iy2 = m*ix2 + c

    int1 = rotate_vector_by_angle_and_shift(np.column_stack([ix1,iy1]), ellipse["alpha"], shiftX=ellipse["xCenter"], shiftY=yc)
    int2 = rotate_vector_by_angle_and_shift(np.column_stack([ix2,iy2]), ellipse["alpha"], shiftX=ellipse["xCenter"], shiftY=yc)

    d1 = compute_rowwise_norm(int1 - np.column_stack([x,y]))
    d2 = compute_rowwise_norm(int2 - np.column_stack([x,y]))
    return np.minimum(d1,d2)
