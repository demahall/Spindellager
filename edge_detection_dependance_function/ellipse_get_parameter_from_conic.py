import numpy as np

def ellipse_get_parameters_from_conic(u, compute_parametric=True, compute_focii_string=True):
    """
    Compute parametric and focii/string parameters of an ellipse from its conic representation.

    Parameters
    ----------
    u : array-like, shape (6,)
        Conic parameters [A, B, C, D, E, F].
        Conic equation: A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0
    compute_parametric : bool, default=True
        Whether to compute the parametric form (center, axes, tilt).
    compute_focii_string : bool, default=True
        Whether to compute the focii and string parameters.

    Returns
    -------
    ellipse : dict
        Dictionary containing:
            xCenter, yCenter, a, b, alpha, F1, F2, s, e, epsilon, u
    """
    u = np.asarray(u, dtype=float)
    if u.shape != (6,):
        raise ValueError("Input u must be a 6-element array.")

    # If focii are requested, parametric must be computed
    if compute_focii_string:
        compute_parametric = True

    # Extract conic parameters (Bronstein convention: divide B, D, E by 2)
    A = u[0]
    B = u[1] / 2
    C = u[2]
    D = u[3] / 2
    E = u[4] / 2
    F_const = u[5]

    # Compute invariants
    Delta = np.linalg.det(np.array([
        [A, B, D],
        [B, C, E],
        [D, E, F_const]
    ]))
    delta = np.linalg.det(np.array([
        [A, B],
        [B, C]
    ]))
    S = A + C

    # Check if this is an ellipse
    if delta > 0 and Delta != 0 and Delta * S < 0:
        if compute_parametric:
            # Center
            xCenter = (B*E - C*D) / delta
            yCenter = (B*D - A*E) / delta

            # Tilt angle
            alpha = 0.5 * np.arctan2(2*B, A - C)

            # Semi-axes
            A_prime = (A + C + np.sqrt((A - C)**2 + 4*B**2)) / 2
            C_prime = (A + C - np.sqrt((A - C)**2 + 4*B**2)) / 2

            axisA = np.sqrt(-Delta / (delta * A_prime))
            axisB = np.sqrt(-Delta / (delta * C_prime))

            a = max(axisA, axisB)
            b = min(axisA, axisB)

            # Adjust angle
            if axisA < axisB:
                alpha += np.pi/2
            else:
                alpha = np.deg2rad(np.mod(np.rad2deg(alpha - np.pi),180))

            if np.abs(alpha - np.pi) < 1e-5:
                alpha = 0.0

            # Linear eccentricity and numeric eccentricity
            e = np.sqrt(a**2 - b**2)
            epsilon = e / a
        else:
            xCenter = yCenter = a = b = alpha = e = epsilon = None

        # Focii and string
        if compute_focii_string and compute_parametric:
            F1 = [xCenter + np.cos(alpha)*e, yCenter + np.sin(alpha)*e]
            F2 = [xCenter - np.cos(alpha)*e, yCenter - np.sin(alpha)*e]
            s = 2*a
        else:
            F1 = F2 = s = None

        # Build result dictionary
        ellipse = dict(
            xCenter=xCenter,
            yCenter=-yCenter if yCenter is not None else None,  # Note: invert Y to match MATLAB
            a=a,
            b=b,
            alpha=alpha,
            F1=F1,
            F2=F2,
            s=s,
            e=e,
            epsilon=epsilon,
            u=u
        )
    else:
        raise ValueError("2nd order curve is not an ellipse (invalid conic parameters).")

    return ellipse
