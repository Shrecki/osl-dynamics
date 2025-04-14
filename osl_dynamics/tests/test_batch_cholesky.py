""" 

Authors: Fabrice Guibert
"""
from osl_dynamics.inference import batched_cov
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
)
import numpy as np


def test_ordering():
    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp

    def get_tril_to_tfp_indices(n_channels):
        """
        Create a direct mapping of indices from np.tril_indices ordering to 
        TensorFlow Probability's FillTriangular ordering.
        
        Parameters
        ----------
        n_channels : int
            Number of channels/dimension of the square matrix.
        
        Returns
        -------
        numpy.ndarray
            Array of indices that can be used to reorder vectors from 
            np.tril_indices ordering to TFP's FillTriangular ordering.
        """
        vec_size = n_channels * (n_channels + 1) // 2
        
        # Create a sequential array for testing
        tril_seq = np.arange(vec_size)
        
        # Create the Cholesky matrix using np.tril_indices ordering
        L = np.zeros((n_channels, n_channels))
        tril_indices = np.tril_indices(n_channels)
        L[tril_indices] = tril_seq
        
        # Vectorize the matrix using TFP's ordering
        tfp_seq = np.zeros(vec_size)
        count = 0
        for row in range(n_channels):
            for col in range(row + 1):
                # This matches TFP's FillTriangular ordering
                # (row-major, lower-triangular, flattened)
                tfp_seq[count] = L[row, col]
                count += 1
        
        # Find mapping from TFP ordering back to np.tril_indices ordering
        mapping = np.zeros(vec_size, dtype=np.int32)
        for i in range(vec_size):
            # Find where each element ended up in the TFP sequence
            mapping[i] = np.where(tfp_seq == i)[0][0]
        
        return mapping

    def reorder_tril_to_tfp_direct(cholesky_vectors, n_channels=None, mapping=None):
        """
        Reorder vectorized Cholesky factors from np.tril_indices ordering 
        to TensorFlow Probability's FillTriangular ordering using direct indexing.
        
        Parameters
        ----------
        cholesky_vectors : numpy.ndarray
            Vectorized Cholesky factors using np.tril_indices ordering.
            Shape is (vec_size, T) where vec_size = n_channels*(n_channels+1)/2
            and T is the number of time points. Can also accept shape (T, vec_size).
        
        n_channels : int, optional
            Number of channels/dimension of the square matrix. 
            Required if mapping is not provided.
        
        mapping : numpy.ndarray, optional
            Precomputed index mapping from get_tril_to_tfp_indices(). 
            If not provided, it will be computed using n_channels.
        
        Returns
        -------
        numpy.ndarray
            Reordered vectorized Cholesky factors in TFP's FillTriangular ordering.
            Shape matches the input shape.
        """
        # Check if we need to transpose
        transpose_needed = False
        if cholesky_vectors.ndim == 2 and cholesky_vectors.shape[0] != cholesky_vectors.shape[1]:
            if cholesky_vectors.shape[1] > cholesky_vectors.shape[0]:
                # Shape is likely (T, vec_size), so transpose for consistent processing
                cholesky_vectors = cholesky_vectors.T
                transpose_needed = True
        
        # Get mapping if not provided
        if mapping is None:
            if n_channels is None:
                raise ValueError("Either mapping or n_channels must be provided")
            mapping = get_tril_to_tfp_indices(n_channels)
        
        # Apply the mapping to reorder the vectors
        if cholesky_vectors.ndim == 1:
            # Single vector
            reordered = cholesky_vectors[mapping]
        else:
            # Multiple vectors
            reordered = cholesky_vectors[mapping, :]
        
        # Transpose back if needed
        if transpose_needed:
            reordered = reordered.T
        
        return reordered

    n_channels = 10

    # Create a test lower triangular matrix
    L = np.zeros((n_channels, n_channels))
    count = 1
    for i in range(n_channels):
        for j in range(i+1):  # Lower triangular
            L[i, j] = count
            count += 1
        
    print("Original lower triangular matrix:")
    print(L)
    
    # Vectorize using np.tril_indices
    tril_indices = np.tril_indices(n_channels)
    vec_tril = L[tril_indices]
    
    print("\nVectorized using np.tril_indices:")
    print(vec_tril)
    
    # Reorder to TFP's ordering
    mapping = get_tril_to_tfp_indices(n_channels)
    vec_tfp = reorder_tril_to_tfp_direct(vec_tril, mapping=mapping)
    
    print("\nReordered for TFP's FillTriangular:")
    print(vec_tfp)
    
    # Verify using TFP's FillTriangular
    fill_triangular = tfp.bijectors.FillTriangular()
    L_reconstructed = fill_triangular(tf.convert_to_tensor([vec_tfp])).numpy()[0]
    
    print("\nMatrix reconstructed by TFP's FillTriangular:")
    print(L_reconstructed)
    
    # Check if reconstruction matches original
    is_correct = np.allclose(L, L_reconstructed)
    print(f"\nReordering is correct: {is_correct}")
    
    # Also test batch reordering
    T = 5
    vec_size = n_channels * (n_channels + 1) // 2
    batch_vectors = np.tile(vec_tril.reshape(-1, 1), (1, T))
    batch_reordered = reorder_tril_to_tfp_direct(batch_vectors, mapping=mapping)
    
    # Check shape
    print(f"\nBatch reordering shape: {batch_reordered.shape} (expected {(vec_size, T)})")
    
    # Check first vector matches previous result
    first_correct = np.allclose(batch_reordered[:, 0], vec_tfp)
    print(f"First batch vector correct: {first_correct}")



def test_sliding_window_covariance():
    n_s = 210570
    n_v = 4
    big_array = np.random.randn(n_s, n_v)
    
    import time
    start = time.time()
    res = batched_cov.batched_covariance_and_cholesky(big_array, 5000, 10000)
    end = time.time()
    print(f"Time: {end-start}")
    cov_chol = res[1]
    assert cov_chol.shape[0] == n_s - 5000 + 1
    assert cov_chol.shape[1] == (n_v+1)*n_v//2
    
    tri_ids = np.tril_indices(n_v)
    
    for i in range(n_s - 5000 +1):
        samples = big_array[i:i+5000]
        if samples.shape[0] < 5000:
            break
        else:
            cholesky_transf = np.linalg.cholesky(np.cov(samples,rowvar=False))[tri_ids[0], tri_ids[1]]
            assert np.allclose(cholesky_transf, cov_chol[i], 1e-4)
