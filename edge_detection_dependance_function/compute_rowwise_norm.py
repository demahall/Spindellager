import numpy as np

"""
%COMPUTEROWWISENORM Helper function to compte the rowwise L2-norm of a n-by-2-matrix
%
%   [ NORMEDVECTOR ] = COMPUTEROWWISENORM( VECTOR )
%
%   Arguments:
%               nby2Matrix:             n-by-2-matrix input
%   output:
%               vectorOfRowwiseNorms:   Vector containing the L2-norm of every row of the input matrix
"""

def compute_rowwise_norm(nby2_matrix):
    """
    Compute the Euclidean (L2) norm of each row of a 2-column matrix.

    Parameters
    ----------
    nby2_matrix : ndarray
        2D NumPy array of shape (N, 2).

    Returns
    -------
    norms : ndarray
        1D array of length N, containing the L2 norm of each row.

    Example
    -------
    >>> M = np.array([[3,4],[5,12]])
    >>> compute_rowwise_norm(M)
    array([ 5., 13.])
    """
    nby2_matrix = np.asarray(nby2_matrix)
    norms = np.linalg.norm(nby2_matrix, axis=1)
    return norms
