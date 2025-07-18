import numpy as np
import cv2

def extract_roi(
    I=None,
    shape='ring',
    xCenter=None,
    yCenter=None,
    ri=None,
    ro=None,
    xLength=None,
    yLength=None
):
    """
    Extracts a region of interest in an image.

    Parameters:
        I (ndarray): Input 2D image.
        shape (str): 'ring' or 'rectangle'.
        xCenter (float): x-coordinate of ROI center.
        yCenter (float): y-coordinate of ROI center.
        ri (float): Inner radius for 'ring'.
        ro (float): Outer radius for 'ring'.
        xLength (float): Rectangle width.
        yLength (float): Rectangle height.

    Returns:
        roiImage: Masked image (same size).
        boundaryImage: Binary boundary mask.
        croppedROI: Cropped ROI image.
        boundingBox: [minX, minY, maxX, maxY].
    """
    # --- Input validation ---
    if I.ndim != 2:
        raise ValueError("Input image must be 2D (grayscale).")
    if shape not in ('ring', 'rectangle'):
        raise ValueError("shape must be 'ring' or 'rectangle'.")
    if xCenter is None or yCenter is None:
        raise ValueError("xCenter and yCenter must be provided.")

    Iuint8 = I.astype(np.uint8)

    ix, iy = Iuint8.shape

    # Create coordinate grid
    x, y = np.meshgrid(
        np.arange(iy) - (xCenter - 1),
        np.arange(ix) - (yCenter - 1)
    )

    # Mask depending on shape
    if shape == 'ring':
        if ri is None or ro is None:
            raise ValueError("ri and ro must be provided for 'ring'.")
        r2 = x**2 + y**2
        mask = (ri**2 <= r2) & (r2 <= ro**2)
    elif shape == 'rectangle':
        if xLength is None or yLength is None:
            raise ValueError("xLength and yLength must be provided for 'rectangle'.")
        mask = (
            (-xLength / 2 <= x) & (x <= xLength / 2) &
            (-yLength / 2 <= y) & (y <= yLength / 2)
        )

    # Extract boundary using bwmorph function like in MATLAB script!!

    def bwmorph(input_matrix):
        output_matrix = input_matrix.copy()

        # Convert to single channel if necessary
        if len(output_matrix.shape) == 3:
            output_matrix = output_matrix[:, :, 0]

        nRows, nCols = output_matrix.shape
        orig = output_matrix.copy()

        for indexRow in range(nRows):
            for indexCol in range(nCols):
                center_pixel = [indexRow, indexCol]
                neighbor_array = neighbors(orig, center_pixel)

                if np.all(neighbor_array):
                    output_matrix[indexRow, indexCol] = 0

        return output_matrix

    def neighbors(input_matrix, input_array):
        (rows, cols) = input_matrix.shape[:2]
        indexRow = input_array[0]
        indexCol = input_array[1]
        output_array = [0] * 4

        # Top neighbor
        output_array[0] = input_matrix[(indexRow - 1) % rows, indexCol]
        # Right neighbor
        output_array[1] = input_matrix[indexRow, (indexCol + 1) % cols]
        # Bottom neighbor
        output_array[2] = input_matrix[(indexRow + 1) % rows, indexCol]
        # Left neighbor
        output_array[3] = input_matrix[indexRow, (indexCol - 1) % cols]

        return output_array

    #Then put mask into bwmorph and we get at the end boundary mask, where interior pixels are removed, leaving only its border
    boundary_image = bwmorph(mask.astype(np.uint8))


    # roiImage: pixels inside ROI, rest = 0
    roiImage = np.zeros_like(Iuint8)
    roiImage[mask] = Iuint8[mask]

    # Compute bounding box
    ys, xs = np.where(boundary_image)
    minX = int(xs.min())
    maxX = int(xs.max())
    minY = int(ys.min())
    maxY = int(ys.max())

    boundingBox = [minX, minY, maxX, maxY]

    # Crop ROI
    croppedROI = roiImage[minY:maxY+1, minX:maxX+1]

    return roiImage, boundary_image, croppedROI, boundingBox
