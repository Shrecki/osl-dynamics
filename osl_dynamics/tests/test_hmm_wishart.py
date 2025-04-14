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
from osl_dynamics.models.hmm_wishart import Config, Model
from osl_dynamics.data import Data as Data

import tensorflow as tf
import keras
import copy
import sys

def test_hmm_wishart_kmeans_init():
    window = 50
    #####
    # Create dataset
    #####
    n_c = 4
    n_s = [4000,14000,19000]
    
    orig_data = [np.random.randn(samples,n_c) for samples in n_s]
    input_data =Data(orig_data,sampling_frequency=1.0)
    input_data.moving_covar_cholesky_vectorized(window)
    
    ####
    # Create model config
    ####
    config = Config(
        n_states=3,
        n_channels=n_c,
        sequence_length=1000,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=1000,
        learn_covariances=True,
        multi_gpu = False,
        window_size = window
    )
    
    model = Model(config)
    
    ####
    # Call k-means initialization and assert no throw
    ####
    model.kmeans_time_course_initialization(input_data)


def test_hmm_wishart_no_crash():
    """When using a Wishart observation model, we don't expect
    to see errors. We will feed a simplified case
    with non-random cholesky factors and assess
    we get indeed all the proper covariances recovered.
    """
    # Initialize unmasked config and model
    # Set model parameters some means and covariances to define currently estimated state by the model

    random_covs = np.zeros((3,10,10))
    
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    import tensorflow_probability as tfp
    random_choleskys = tfp.math.fill_triangular_inverse(np.linalg.cholesky(random_covs)) #[:,idx[0],idx[1]]
    # Create now fake data!
    
    from scipy.stats import wishart as wishart_scipy
    
    fake_data = np.zeros((3000, 55))
    fake_data[:1000] = tfp.math.fill_triangular_inverse(np.linalg.cholesky(wishart_scipy(df=90, scale = random_covs[0]).rvs(1000)))
    fake_data[1000:2000] = tfp.math.fill_triangular_inverse(np.linalg.cholesky(wishart_scipy(df=90, scale = random_covs[1]).rvs(1000)))
    fake_data[2000:] = tfp.math.fill_triangular_inverse(np.linalg.cholesky(wishart_scipy(df=90, scale = random_covs[2]).rvs(1000)))
    
        
    from sklearn.cluster import KMeans
    
    kmeans_model = KMeans(3)
    kmeans_model.fit(fake_data)
            
    recovered_cholesky = tfp.math.fill_triangular(kmeans_model.cluster_centers_)
    recovered_covs = tf.linalg.matmul(recovered_cholesky, recovered_cholesky, transpose_b=True)

    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=1000,
        learn_covariances=True,
        initial_covariances=recovered_covs.numpy(),
        multi_gpu = False,
        window_size = 90
    )
    
    model = Model(config)
    print(model.summary())
    
    data = Data([fake_data],time_axis_first=True, sampling_frequency=250.0)

    model.fit(data.dataset(1000,30))
    
    probas =  model.get_alpha(data)
    
    state_0 = np.argmax(probas[:1000].mean(axis=0))
    state_1 = np.argmax(probas[1000:2000].mean(axis=0))
    state_2 = np.argmax(probas[2000:].mean(axis=0))
    
    print(probas)
    
    if(np.any(np.isnan(probas))):
        print(model.get_covariances())
    
    assert state_0 != state_1
    assert state_1 != state_2

    assert np.allclose(probas[:1000,state_0], 1)
    assert np.allclose(probas[1000:2000,state_1], 1)
    assert np.allclose(probas[2000:,state_2], 1)


def test_hmm_wishart_synthetic_dataset():
    """
    A test where we generate a dataset from a mixture of multivariate gaussians.
    We then compute a sliding-window covariance matrix from the data (using the "prepare" method)
    and feed it to the model.    
    """
    
    data = Data([np.random.randn(50000,10)],sampling_frequency=1.0)
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=1000,
        learn_covariances=True,
        compute_cov_runtime=True,
        multi_gpu = False,
        window_size = 90
    )
    
    model = Model(config)
    print(model.summary())
    
    model.fit(data)
    
    
    # Generate a list of random state changes.
    # We start from a purely ordered list and slightly perturb it.
    # Expectation: window size will remove instantaneous observations,
    # leading to smooth state timecourses
    
    # Generate MV gaussian samples for each state
    
    # Assign to time series
    
    # Transform as vectorized Cholesky of sliding-window covars
    
    # Initialize model and train (we hope for no errors!)