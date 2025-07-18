import numpy as np

def make_binary_image_from_points(image_size, pixel_coordinates):
    """
    Creates a binary image of specified size from pixel coordinates.

    Parameters:
        image_size: tuple of (height, width)
        pixel_coordinates: (N,2) array of (x,y) positions

    Returns:
        binary_image: boolean array of shape image_size
    """
    # Argument check
    if not isinstance(image_size, (tuple, list, np.ndarray)) or len(image_size) != 2:
        raise ValueError("image_size must be a tuple or list with 2 elements (height, width).")
    if np.any(np.array(image_size) <= 0):
        raise ValueError("image_size must contain positive integers.")
    if pixel_coordinates.shape[1] != 2:
        raise ValueError("pixel_coordinates must be an (N,2) array.")

    H, W = int(image_size[0]), int(image_size[1])

    # Initialize binary image
    binary_image = np.zeros((H, W), dtype=bool)

    # Round and remove duplicates
    coords = np.round(pixel_coordinates).astype(int)
    coords = np.unique(coords, axis=0)

    # Remove out-of-bounds
    valid_mask = (
        (coords[:, 0] >= 0) & (coords[:, 0] < W) &
        (coords[:, 1] >= 0) & (coords[:, 1] < H)
    )
    coords = coords[valid_mask]

    # Set pixels
    binary_image[coords[:,1], coords[:,0]] = 1

    return binary_image
