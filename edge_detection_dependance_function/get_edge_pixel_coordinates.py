import numpy as np

def get_edge_pixel_coordinates(edge_image):
    """
    Compute the coordinates of the edge pixels in a binary image.

    Parameters
    ----------
    edge_image : ndarray
        2D boolean NumPy array.
        Pixels with True are considered edge pixels.

    Returns
    -------
    x : ndarray
        1D array of x-coordinates (column indices).
    y : ndarray
        1D array of y-coordinates (row indices negated, so y-axis points upwards).

    Notes
    -----
    In image coordinates, the top-left pixel has (row=0, col=0).
    Negating the row indices converts to a Cartesian coordinate system where y increases upward.
    """
    rows, cols = np.nonzero(edge_image)
    x = cols
    y = -rows
    return x, y
