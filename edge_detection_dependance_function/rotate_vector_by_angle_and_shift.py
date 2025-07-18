import numpy as np


def rotate_vector_by_angle_and_shift(vector, angle, shiftX=0, shiftY=0):

    """
    Rotate 2D vectors by angle and shift.

    vector: (N,2) or (2,N) array
    angle: radians
    Returns same shape as input.
    """

    vector = np.asarray(vector)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a, cos_a]])

    if vector.shape[0] == 2 and vector.shape[1] != 2:
        # shape (2,N)
        rotated = R @ vector
        rotated[0, :] += shiftX
        rotated[1, :] += shiftY
    else:
        # shape (N,2)
        rotated = (R @ vector.T).T
        rotated[:, 0] += shiftX
        rotated[:, 1] += shiftY

    return rotated
