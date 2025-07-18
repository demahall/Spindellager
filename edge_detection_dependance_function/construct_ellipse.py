import numpy as np

"""
%CONSTRUCTELLIPSE Computes the points of an ellipse for given center point, semi-axes and tilt angle
%
%   [ XELLIPSE, YELLIPSE ] = CONSTRUCTELLIPSE( XCENTER, YCENTER, A, B, ALPHA )
%
%   Arguments:
%               xCenter:    x-coordinate of center
%               yCenter:    y-coordinate of center
%               a:          semi-major axis 
%               b:          semi-minor axis 
%               alpha:     tilt angle
%   output:
%               xEllipse:    Vector containing the x-coordinates of ellipse points
%               yEllipse:    Vector containing the y-coordinates of ellipse points
"""

def construct_ellipse(x_center, y_center, a, b, alpha):

    phi = np.arange(0, 2 * np.pi, 0.0001)
    # Unrotated ellipse
    x = a * np.cos(phi)
    y = b * np.sin(phi)
    # Rotation matrix components
    cos_a = np.cos(-alpha)
    sin_a = np.sin(-alpha)
    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y
    # Translate
    x_final = x_rot + x_center
    y_final = y_rot + y_center
    return x_final, y_final
