import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
)

import osl_dynamics
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.data import Data
from osl_dynamics.simulation import HMM_MVN

import keras

def test_ll_mask_none_equals_no_mask():
    """The log likelihood of a data under 
       the observation model should be exactly the same
       when we consider the fully masked data,
       the data with a :code:`None` log likelihood mask,
       and the data under config without masking.
    """
    # Initialize unmasked config and model
    # Set model parameters some means and covariances to define currently estimated state by the model

    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    random_means = np.random.randn(3,10)
        
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=False,
        learn_covariances=False,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=1,
        multi_gpu = False,
        use_mask= False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T
    )
    
    model = Model(config)


    data = Data([np.random.randn(10000, 10)],time_axis_first=True, sampling_frequency=250.0)
    
    dataset = model.make_dataset(data, concatenate=True)

    # Get likelihood
    ll_base = []
    for data in dataset:
        x = data["data"]
        ll_1 = model.get_likelihood(x)
        ll_base.append(ll_1)
    ll_base = np.concatenate(ll_base)

    # Required to free resources and be able to run multiple models back to back
    del model
    keras.backend.clear_session()
    
    # Now, same but with a masking approach and mask = None
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=False,
        learn_covariances=False,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=1,
        multi_gpu = False,
        use_mask= True,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T
    )
    
    model = Model(config)
    
    ll_mask_none = []
    for data in dataset:
        x = data["data"]
        ll_1 = model.get_likelihood(x)
        ll_mask_none.append(ll_1)
        
    ll_mask_none = np.concatenate(ll_mask_none)
    
    # Required to free resources and be able to run multiple models back to back
    del model
    keras.backend.clear_session()
    assert np.allclose(ll_mask_none,ll_base), f"Expected {ll_base}, actual {ll_mask_none}"
    
def test_ll_mask_full_equals_no_mask():
    """The log likelihood of a data under 
       the observation model should be exactly the same
       when we consider the fully masked data,
       the data with a :code:`None` log likelihood mask,
       and the data under config without masking.
    """
    # Initialize unmasked config and model
    # Set model parameters some means and covariances to define currently estimated state by the model

    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    random_means = np.random.randn(3,10)
        
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=False,
        learn_covariances=False,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=1,
        multi_gpu = False,
        use_mask= False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T
    )
    
    model = Model(config)


    data = Data([np.random.randn(10000, 10)],time_axis_first=True, sampling_frequency=250.0)
    
    dataset = model.make_dataset(data, concatenate=True)

    # Get likelihood
    ll_base = []
    for data in dataset:
        x = data["data"]
        ll_1 = model.get_likelihood(x)
        ll_base.append(ll_1)
    ll_base = np.concatenate(ll_base)

    # Required to free resources and be able to run multiple models back to back
    del model
    keras.backend.clear_session()
    
    # Now, same but with a masking approach and mask = None
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=False,
        learn_covariances=False,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=1,
        multi_gpu = False,
        use_mask= True,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T
    )
    
    model = Model(config)
    
    mask = np.ones(1000,dtype=bool)
    
    ll_mask_none = []
    for data in dataset:
        x = data["data"]
        ll_1 = model.get_likelihood(x,mask)
        ll_mask_none.append(ll_1)
        
    ll_mask_none = np.concatenate(ll_mask_none)
    
    # Required to free resources and be able to run multiple models back to back
    del model
    keras.backend.clear_session()
    assert np.allclose(ll_mask_none,ll_base), f"Expected {ll_base}, actual {ll_mask_none}"

def test_ll_under_mask_equals_no_mask():
    """The log likelihood of the considered points
    must be exactly equal to the log likelihood of the non masked data
    """
    assert True == False

def test_ll_out_mask_zero():
    """The log likelihood outside of the LL mask
    must be exactly zero.
    """
    assert True == False

def test_fit_with_mask_equivalent_to_segmenting():
    """Segmenting out points should yield qualitatively similar results
    as masking them.
    """
    assert True == False