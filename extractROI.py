import numpy as np


def extract_roi(image, outer_center, outer_radius, inner_center, inner_radius):
    """
    Extract an annular region (between two circles) from the image.

    Parameters:
        image: 2D numpy array (grayscale image)
        outer_center: tuple (x, y) of outer circle center
        outer_radius: float, radius of outer circle
        inner_center: tuple (x, y) of inner circle center
        inner_radius: float, radius of inner circle

    Returns:
        masked_image: The same shape as input, pixels outside annulus set to 0
        cropped_image: Cropped bounding rectangle containing the annulus
    """
    h, w = image.shape

    # Create a grid of x and y coordinates
    Y, X = np.ogrid[:h, :w]

    # Compute distance from each pixel to outer center
    dist_outer = np.sqrt((X - outer_center[0]) ** 2 + (Y - outer_center[1]) ** 2)

    # Compute distance from each pixel to inner center
    dist_inner = np.sqrt((X - inner_center[0]) ** 2 + (Y - inner_center[1]) ** 2)

    # Build the mask: pixels between inner and outer circle
    mask = (dist_outer <= outer_radius) & (dist_inner >= inner_radius)


    # Create masked image
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    # Compute bounding rectangle around the outer circle
    x_min = max(int(outer_center[0] - outer_radius), 0)
    x_max = min(int(outer_center[0] + outer_radius), w)
    y_min = max(int(outer_center[1] - outer_radius), 0)
    y_max = min(int(outer_center[1] + outer_radius), h)



    return masked_image,(x_min,y_min)