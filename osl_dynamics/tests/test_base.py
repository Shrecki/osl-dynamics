"""
Authors: Fabrice Guibert

"""

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
)

import osl_dynamics
from osl_dynamics.data import Data as Data

import tensorflow as tf
import tensorflow_probability as tfp
import keras
import copy
import sys


def test_sliding_window_incorrect_channels_spec():
    """
    Verify that using wrong window size raises an error
    """
    n_c = 4
    n_s = [1000,1400,900]
    
    orig_data = [np.random.randn(samples,n_c) for samples in n_s]
    input_data = Data(orig_data,sampling_frequency=1.0,time_axis_first=True)
    with pytest.raises(ValueError):
        input_data.moving_covar_cholesky_vectorized("boop")
    with pytest.raises(ValueError):
        input_data.moving_covar_cholesky_vectorized(None)
    with pytest.raises(ValueError):
        input_data.moving_covar_cholesky_vectorized(-1)
    with pytest.raises(ValueError):
        input_data.moving_covar_cholesky_vectorized(0)
    with pytest.raises(ValueError):
        input_data.moving_covar_cholesky_vectorized(3)
    with pytest.raises(ValueError):
        input_data.moving_covar_cholesky_vectorized(10.0)
    input_data.prepare({'moving_covar_cholesky_vectorized': {'n_window': 10}})

def test_sliding_window_covar_correct():
    """
    Verify that computed sliding window covariance is correct
    """
    
    # Construct a random dataset
    n_c = 4
    n_s = [1000,1400,900]
    
    orig_data = [np.random.randn(samples,n_c) for samples in n_s]
    
    window_sizes = [10, 15,50]
    
    for window in window_sizes:
        # Generate data
        input_data = Data(orig_data,sampling_frequency=1.0,time_axis_first=True)
        
        # Compute naive covar with specified window
        expected_covars = []
        for a in orig_data:
            n_t = a.shape[0]
            n_samples = n_t - window + 1
            cov_arr = np.zeros((n_samples, n_c,n_c))
            for i in range(n_samples):
                c = np.cov(a[i:i+window],rowvar=False)
                cov_arr[i] = c
            # Compute Cholesky and vectorize
            expected_covars.append(tfp.math.fill_triangular_inverse(np.linalg.cholesky(cov_arr)))
        
        # Prepare data
        input_data.moving_covar_cholesky_vectorized(window)
        
        # Compare up to errors
        for i in range(len(input_data.arrays)):
            assert np.allclose(input_data.arrays[i], expected_covars[i],atol=1e-4)
        
        # Same for batched FFT    
        input_data = Data(orig_data,sampling_frequency=1.0,time_axis_first=True)
        input_data.moving_covar_cholesky_vectorized(window,approach="batch_fft")
        
        for i in range(len(input_data.arrays)):
            assert np.allclose(input_data.arrays[i], expected_covars[i],atol=1e-4)
            
        input_data = Data(orig_data,sampling_frequency=1.0,time_axis_first=True)
        input_data.moving_covar_cholesky_vectorized(window,approach="naive")
        
        for i in range(len(input_data.arrays)):
            assert np.allclose(input_data.arrays[i], expected_covars[i],atol=1e-4)
            
def test_different_sliding_windows_produce_different_covars():
    """
    Verify that different sliding windows do generate different covariances and
    not some trivial result.
    """
    
    # Construct a random dataset
    n_c = 4
    n_s = [1000,1400,900]
    
    orig_data = [np.random.randn(samples,n_c) for samples in n_s]
    
    input_data_a = Data(orig_data,sampling_frequency=1.0,time_axis_first=True)
    input_data_a.moving_covar_cholesky_vectorized(10)
    
    input_data_b = Data(orig_data,sampling_frequency=1.0,time_axis_first=True)
    input_data_b.moving_covar_cholesky_vectorized(15)
    # Compare up to errors
    for i in range(len(input_data_b.arrays)):
        assert not np.allclose(input_data_a.arrays[i][:800], input_data_b[i][:800],atol=1e-4)
                    
def test_channel_nbr_correct():
    """
    Verify that computed number of channels is correct
    """
    n_c = 4
    n_s = [1000,1400,900]
    
    orig_data = [np.random.randn(samples,n_c) for samples in n_s]
    input_data = Data(orig_data,sampling_frequency=1.0,time_axis_first=True)
    input_data.moving_covar_cholesky_vectorized(30)
    
    assert input_data.n_covar_channels == int(4*(4+1)/2)