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


def test_sliding_window_covariance():
    n_s = 210570
    n_v = 10
    big_array = np.random.randn(n_s, n_v)
    
    res = batched_cov.batched_covariance_and_cholesky(big_array, 5000, 10000)
    
    cov_chol = res[1]
    assert cov_chol.shape[0] == n_s - 5000 + 1
    assert cov_chol.shape[1] == (n_v+1)*n_v//2
    
    tri_ids = np.tril_indices(10)
    
    for i in range(n_s - 5000 +1):
        samples = big_array[i:i+5000]
        if samples.shape[0] < 5000:
            break
        else:
            cholesky_transf = np.linalg.cholesky(np.cov(samples,rowvar=False))[tri_ids[0], tri_ids[1]]
            assert np.allclose(cholesky_transf, cov_chol[i], 1e-4)
