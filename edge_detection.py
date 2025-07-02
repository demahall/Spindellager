import numpy as np
from skimage import feature, morphology, measure

def detect_edges(
    image,
    threshold=(0.1, 0.3),
    sigma=1.0,
    min_area=30,
    mask=None
):
    """
    Perform Canny edge detection and clean up edges.

    Parameters:
        image: 2D numpy array, grayscale image.
        threshold: tuple of two floats (low, high) between 0 and 1.
        sigma: float, Gaussian smoothing sigma.
        min_area: int, minimum area of connected components.
        mask: optional binary mask to remove edges in ROI boundaries.

    Returns:
        cleaned_edges: binary image with cleaned edges.
    """
    # Normalize image to 0..1 if needed
    if np.issubdtype(image.dtype, np.integer):
        image = image / 255.0

    # Canny edge detection
    edges = feature.canny(image, sigma=sigma, low_threshold=threshold[0], high_threshold=threshold[1])

    # Remove edges overlapping the mask if provided
    if mask is not None:
        edges = edges & ~mask

    # Remove small edge regions
    cleaned_edges = morphology.remove_small_objects(edges, min_size=min_area)

    return cleaned_edges
