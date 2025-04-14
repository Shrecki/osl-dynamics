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
from osl_dynamics.inference import batched_cov

import tensorflow as tf
import tensorflow_probability as tfp
import keras
import copy
import sys

def test_sliding_window_cov_naive():
    n_c = 4
    n_s = [1000]
    
    orig_data = [np.random.randn(samples,n_c) for samples in n_s] 
    
    import time
    
    start = time.time()
    for i in range(2):
        input_data = Data(orig_data,sampling_frequency=1.0,time_axis_first=True)
        
        # Naive preparation
        input_data.prepare({'moving_covar_cholesky_vectorized': {'n_window': 10, 'approach':'naive'}})
    end = time.time()
    print((end-start)/2)
    
def test_sliding_window_cov_optim():
    n_c = 4
    n_s = [1000]
    
    orig_data = [np.random.randn(samples,n_c) for samples in n_s] 
    import time
    start = time.time()

    out = batched_cov.batched_covariance_and_cholesky(orig_data[0],10,1000)
    end = time.time()
    
    print(f"Opti: {end-start}")

    input_data = Data(orig_data,sampling_frequency=1.0,time_axis_first=True)
        
    # Naive preparation
    start = time.time()
    input_data.prepare({'moving_covar_cholesky_vectorized': {'n_window': 10, 'approach':'naive'}})
    end = time.time()
    print(f"Naive: {end-start}")

    
    print(out[1].shape)
    print(out[1][0])
    print(np.linalg.cholesky(np.cov(orig_data[0][:10],rowvar=False)))
    print(input_data.arrays[0][0])
    #assert np.allclose(out[1], input_data.arrays[0], atol=1e-4)

    

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
        input_data.moving_covar_cholesky_vectorized(window,approach="batch_fft",batch_size=1000)
        
        for i in range(len(input_data.arrays)):
            assert np.allclose(input_data.arrays[i], expected_covars[i],atol=1e-4)
            
        input_data = Data(orig_data,sampling_frequency=1.0,time_axis_first=True)
        input_data.moving_covar_cholesky_vectorized(window,approach="naive")
        
        for i in range(len(input_data.arrays)):
            assert np.allclose(input_data.arrays[i], expected_covars[i],atol=1e-4)
            
        input_data = Data(orig_data,sampling_frequency=1.0,time_axis_first=True)
        input_data.moving_covar_cholesky_vectorized(window,approach="batch_cython", batch_size=1000)
        

        #print(input_data.arrays[0])
        #print(expected_covars[0])
        for i in range(len(input_data.arrays)):
            assert np.allclose(input_data.arrays[i], expected_covars[i],atol=1e-4)
        
        
def test_func_array_trim():
    big_array = np.zeros((210570, 10))
    
    def fill_array(x, M,B):
        T = x.shape[0]
        N = x.shape[1]
        L_total = T - M + 1
        vec_size = (N*(N+1))//2

        # Preallocate final output arrays
        covs_out = np.zeros((L_total, vec_size), dtype=np.float64)
        chols_out = np.zeros((L_total, N*(N+1)//2), dtype=np.float64)

        # Overlap between batches to ensure continuous coverage
        O = M - 1
        effective_batch_size = B - O  # The number of unique samples per batch after accounting for overlap
        
        # Process data in batches, but ensure we generate outputs for every valid window
        out_index = 0  # Position in the output arrays
        
        batch_start = 0  # Start index in the original array

        while batch_start < T - M + 1:  # Continue as long as we can form at least one valid window
            # Calculate the end of this batch (limited by array size)
            batch_end = min(batch_start + B, T)
            
            # Skip this batch if it doesn't have enough elements for a full window
            if batch_end - batch_start < M:
                print("Skipped")
                break
                
            # Extract the current batch (view, not copy)
            print(f"Window considered goes from {batch_start} to  {batch_end}")
            batch = x[batch_start:batch_end]
            
            print(f"batch shape: {batch.shape}")
            
            # Compute covariances for this batch
            batch_cov_result = np.ones((batch.shape[0]-M+1,N*(N+1)//2)) #compute_cov_fft_vectorized(batch, M)
            
            print(f"batch cov res shape: {batch_cov_result.shape}")

            # Number of valid windows in this batch
            L_batch = batch_cov_result.shape[0]
            
            # Determine which outputs to keep from this batch
            valid_start = 0
            valid_end =  L_batch
            print(f"valid start: {valid_start} valid_end: {valid_end}")
            # Ensure valid_start doesn't exceed valid_end
            valid_start = min(valid_start, valid_end)
            
            # Calculate number of valid windows
            num_valid = valid_end - valid_start
            
            print(f"Number of valid entries: {num_valid}")
            if num_valid > 0:
                # Only process if there are valid windows
                batch_valid_cov = batch_cov_result[valid_start:valid_end]
                
                # Compute Cholesky decomposition
                batch_chol_result = batch_valid_cov#compute_cholesky(batch_valid_cov, N)
                
                # Copy results to output arrays
                covs_out[out_index:out_index+num_valid] = batch_valid_cov
                chols_out[out_index:out_index+num_valid] += batch_chol_result
                
                # Update output index
                out_index += num_valid
            
            # Move to next batch, advancing by effective_batch_size
            batch_start += effective_batch_size
        
        # If we didn't fill the entire output arrays, trim them
        if out_index < L_total:
            return covs_out[:out_index], chols_out[:out_index]
        else:
            return covs_out, chols_out
        
    res = fill_array(big_array, 5000, 10000)
    assert res[1].shape[0] == big_array.shape[0]-5000+1
    print(np.unique(res[1]))
            
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