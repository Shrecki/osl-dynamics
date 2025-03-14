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
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.data import Data as Data
from osl_dynamics.simulation import HMM_MVN

import tensorflow as tf
import keras
import copy
import sys

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
    
    mask = np.ones(10000,dtype=bool)
    mask_dataset = model.make_dataset(mask, concatenate=True)
    
    combined_dataset = tf.data.Dataset.zip((dataset, mask_dataset))
    
    ll_mask_none = []
    i = 0
    
    for data_item, mask_item in combined_dataset:
        x = data_item["data"]
        m = mask_item["data"]
        ll_1 = model.get_likelihood(x,m)
        ll_mask_none.append(ll_1)
        i+=1
    ll_mask_none = np.concatenate(ll_mask_none)
    
    # Required to free resources and be able to run multiple models back to back
    del model
    keras.backend.clear_session()
    assert np.allclose(ll_mask_none,ll_base), f"Expected {ll_base}, actual {ll_mask_none}"

def test_ll_under_mask_equals_no_mask():
    """The log likelihood of the considered points
    must be exactly equal to the log likelihood of the non masked data
    """
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
    
    mask = np.ones(10000,dtype=bool)
    mask[400:700] = 0
    mask[1350:1781] = 0
    mask_dataset = model.make_dataset(mask, concatenate=True)
    
    combined_dataset = tf.data.Dataset.zip((dataset, mask_dataset))
    
    ll_mask_none = []
    i = 0
    
    for data_item, mask_item in combined_dataset:
        x = data_item["data"]
        m = mask_item["data"]
        ll_1 = model.get_likelihood(x,m)
        ll_mask_none.append(ll_1)
        i+=1
    ll_mask_none = np.concatenate(ll_mask_none)
    
    # Required to free resources and be able to run multiple models back to back
    del model
    keras.backend.clear_session()
    
    assert np.allclose(ll_mask_none[:,mask],ll_base[:,mask])
    assert not np.allclose(ll_mask_none, ll_base)
    #assert np.allclose(ll_mask_none,ll_base), f"Expected {ll_base}, actual {ll_mask_none}"
    
def test_likelihood_out_mask_one():
    """The log likelihood outside of the LL mask
    must be exactly one.
    """
    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    random_means = np.random.randn(3,10)
        
    
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
    
    data = Data([np.random.randn(10000, 10)],time_axis_first=True, sampling_frequency=250.0)    
    dataset = model.make_dataset(data, concatenate=True)
    
    
    mask = np.ones(10000,dtype=bool)
    mask[400:700] = 0
    mask[1350:1781] = 0
    mask_dataset = model.make_dataset(mask, concatenate=True)
    
    
    combined_dataset = tf.data.Dataset.zip((dataset, mask_dataset))
    
    ll_mask_none = []
    i = 0
    
    for data_item, mask_item in combined_dataset:
        x = data_item["data"]
        m = mask_item["data"]
        ll_1 = model.get_likelihood(x,m)
        ll_mask_none.append(ll_1)
        i+=1
    ll_mask_none = np.concatenate(ll_mask_none)
    
    # Required to free resources and be able to run multiple models back to back
    del model
    keras.backend.clear_session()
    assert np.allclose(ll_mask_none[:,~mask],1)
    
def test_posterior_marginal_masked():
    ########
    # Setup model parameters
    ########
    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    
        
    random_means = np.random.randn(3,10)
    
    ##########
    # Generate data
    ##########
    
    data_ = np.random.randn(10000, 10)
    data = Data([data_],time_axis_first=True, sampling_frequency=250.0)
    mask = np.ones(10000,dtype=bool)
    mask[400:700] = 0
    mask[1350:1781] = 0
    
    
    #########
    # Setup first model
    #########
    
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask= True,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    
    dataset = model.make_dataset(data, shuffle=False, concatenate=True)
    mask_dataset = model.make_dataset(mask, shuffle=False, concatenate=True)
    # Set static loss scaling factor (Must be done before fusing)
    model.set_static_loss_scaling_factor(dataset)
    # Fuse as a single dataset
    dataset = tf.data.Dataset.zip((dataset, mask_dataset))
    
    ##########
    # Get posteriors for the model on full data with censored LL
    ##########
    
    gammas_censored = []
    xis_masked = []
    lls_censored = []
    for element in dataset:
        if isinstance(element, (list, tuple)) and len(element) == 2:
            data, ll_masks = element
            ll_masks = ll_masks["data"]
        else:
            data, ll_masks = element, None
        x = data["data"]
        lls_censored.append(model.get_likelihood(x,ll_masks=ll_masks))

        gamma, xi = model.get_posterior(x,ll_masks=ll_masks)
        gammas_censored.append(gamma)
        xis_masked.append(xi)
        
    del model
    keras.backend.clear_session()
        
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask= False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
    )
    
    model = Model(config)
    data = Data([data_[mask]],time_axis_first=True, sampling_frequency=250.0)
    dataset = model.make_dataset(data, shuffle=False, concatenate=True)
    # Set static loss scaling factor (Must be done before fusing)
    model.set_static_loss_scaling_factor(dataset)
        
    #########
    # Get gammas for model of cropped datapoints
    #########
    gammas_cropped = []
    xis_ = []
    lls_cropped = []
    for element in dataset:
        if isinstance(element, (list, tuple)) and len(element) == 2:
            data, ll_masks = element
            ll_masks = ll_masks["data"]
        else:
            data, ll_masks = element, None
        x = data["data"]
        # Shape of gamma is (batch_size*sequence_length, n_states)
        # Shape of xi is (batch_size*sequence_length-1, n_states*n_states)
        gamma, xi = model.get_posterior(x,ll_masks=ll_masks)
        gammas_cropped.append(gamma)
        lls_cropped.append(model.get_likelihood(x,ll_masks=None))
        xis_.append(xi)
    gammas_cropped = np.concatenate(gammas_cropped)

    del model
    keras.backend.clear_session()

    ########
    # Because censored and uncensored have different shapes due to batching and masking:
    # -> Apply the mask to the censored LL approach
    # -> Crop result to fit in batch size
    ########
    gammas_censored = np.concatenate(gammas_censored)
    gammas_censored_cropped = gammas_censored[mask]
    n_s = gammas_censored_cropped.shape[0] // 1000 * 1000
    gammas_censored_cropped = gammas_censored_cropped[:n_s]
    
    lls_censored_cropped = np.concatenate(lls_censored)[:,mask]
    lls_censored_cropped = lls_censored_cropped[:,:n_s]

    #####
    # Compare entries.
    # Places where we expect to see differences are boundaries, which would drive effects in differences.
    # The segments are [0,400], [700:1350], [1781:]
    # Consequently, we should expect that [0,400] is untouched, as there is no censoring there
    # Due to potential boundary effects of Baum Welch around edges (forward-backward pass), 
    # we must take a small edge off to compare, hence why we consider samples [0,390] instead of [0,400]
    #####
    assert np.allclose(gammas_censored_cropped[:390],gammas_cropped[:390])
    
    # We also verify here that in this segment, the shifts in data is not explained only by state transitions
    # but also by observations
    assert not np.allclose(gammas_censored_cropped[1:391], gammas_censored_cropped[:390] @ random_T)
    
    
    ####
    # In the samples which are censored (t=400 to 699), the posterior is driven only by the transition matrix
    ####
    assert np.allclose(gammas_censored[400:600], gammas_censored[399:599] @ random_T)
    
def test_fit_with_ll_masks_on_default_raises_ValueError():
    """
    Calling fit on a model that does not support masks must throw a value error
    """    
    ########
    # Setup model parameters
    ########
    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    
        
    random_means = np.random.randn(3,10)
    
    ##########
    # Generate data
    ##########
    
    data_ = np.random.randn(10000, 10)
    data = Data([data_],time_axis_first=True, sampling_frequency=250.0)
    mask = np.ones(10000,dtype=bool)
    mask[400:700] = 0
    mask[1350:1781] = 0
    
    
    ####
    # Case 1: default model
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    
    with pytest.raises(ValueError) as e:
        h = model.fit(data,ll_masks=mask)
    assert "Cannot use ll_masks in this config. Set use_mask=True in your config." in str(e.value)
    del model
    keras.backend.clear_session()
    
    
    ####
    # Case 2: semi supervised model without states passed
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask=False,
        semi_supervised=True,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    
    with pytest.raises(ValueError) as e:
        h = model.fit(data,ll_masks=mask)
    assert "Cannot use ll_masks in this config. Set use_mask=True in your config." in str(e.value)
    del model
    keras.backend.clear_session()
    
    
    ####
    # Case 3: semi supervised model WITH states passed
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask=False,
        semi_supervised=True,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    state_seq = np.ones(1000*30)*(-1)

    with pytest.raises(ValueError) as e:
        h = model.fit(data,ll_masks=mask,forced_states=state_seq)
    assert "Cannot use ll_masks in this config. Set use_mask=True in your config." in str(e.value)
    del model
    keras.backend.clear_session()
    
def test_fit_with_ll_state_seq_without_option_raises_ValueError():
    """
    Calling fit and providing state_seq on a model that does not support state sequences must throw a value error
    """    
    ########
    # Setup model parameters
    ########
    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    state_seq = np.ones(1000*30)*(-1)

        
    random_means = np.random.randn(3,10)
    
    ##########
    # Generate data
    ##########
    
    data_ = np.random.randn(10000, 10)
    data = Data([data_],time_axis_first=True, sampling_frequency=250.0)
    mask = np.ones(10000,dtype=bool)
    mask[400:700] = 0
    mask[1350:1781] = 0
    
    
    ####
    # Case 1: default model
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    
    with pytest.raises(ValueError) as e:
        h = model.fit(data,forced_states=state_seq)
    assert "Cannot use forced_states in this config. Set semi_supervised=True in your config." in str(e.value)
    del model
    keras.backend.clear_session()
    
    
    ####
    # Case 2: masked model without mask passed
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask=True,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    
    with pytest.raises(ValueError) as e:
        h = model.fit(data,forced_states=state_seq)
    assert "Cannot use forced_states in this config. Set semi_supervised=True in your config." in str(e.value)
    del model
    keras.backend.clear_session()
    
    
    ####
    # Case 3: masked model WITH mask passed
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask=True,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)

    with pytest.raises(ValueError) as e:
        h = model.fit(data,ll_masks=mask,forced_states=state_seq)
    assert "Cannot use forced_states in this config. Set semi_supervised=True in your config." in str(e.value)
    del model
    keras.backend.clear_session()
    
    
def test_fit_mask_slight_differences():
    """
    Starting from the same dataset,
    fitting with masked likelihood MUST
    yield differences compared to fitting full
    likelihood with respect to recovered parameters.
    """
    ########
    # Setup model parameters
    ########
    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    
        
    random_means = np.random.randn(3,10)
    
    ##########
    # Generate data
    ##########
    
    data_ = np.random.randn(10000, 10)
    data = Data([data_],time_axis_first=True, sampling_frequency=250.0)
    mask = np.ones(10000,dtype=bool)
    mask[400:700] = 0
    mask[1350:1781] = 0
    
    
    #########
    # Setup first model
    #########
    
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask= True,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    h = model.fit(data,ll_masks=mask)
    means_censored = model.get_means()
    del model
    keras.backend.clear_session()
        
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask= False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
    )
    
    model = Model(config)
    
    data = Data([data_],time_axis_first=True, sampling_frequency=250.0)
    h = model.fit(data)
    means_full = model.get_means()

    del model
    keras.backend.clear_session()
    
    assert np.any(np.abs(means_censored - means_full) > 1e-6)
    
    
    
def test_override_gamma_with_None_is_idop():
    """
    When the state sequence is None, we expect to get back exactly original gamma
    """
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask= True,
        
    )
    
    model = Model(config)
    
    gamma = np.random.randn(1000*30, 5)
    gamma_corr = model.override_gamma(copy.deepcopy(gamma), None)
    
    # We assume EXACT equality
    assert np.all(gamma ==gamma_corr)
    
def test_override_gamma_with_NegStateSeq_isNoOP():
    """
    When the state sequence is negative, we expect to get back exactly original gamma
    """
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask= True,
        
    )
    
    model = Model(config)
    
    gamma = np.random.randn(1000*30, 5)
    state_seq = np.ones(1000*30)*(-1)
    gamma_corr = model.override_gamma(copy.deepcopy(gamma), state_seq)
    
    # We assume EXACT equality
    assert np.all(gamma ==gamma_corr)
    
def test_override_gamma_proper():
    """
    When the state sequence is non-negative, we expect to get back hard state assignments
    within assigned states
    """
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask= True,
        
    )
    
    model = Model(config)
    
    gamma = np.random.randn(1000*30, 3)
    state_seq = np.ones(1000*30,dtype=int)*(-1)
    state_seq[200:250] = 0
    state_seq[400:860] = 2
    state_seq[900:920] = 1
    
    gamma_corr = model.override_gamma(copy.deepcopy(gamma), state_seq)
    
    # We assume EXACT equality outside corrected gammas
    unmodified_mask = state_seq < 0
    assert np.all(gamma[unmodified_mask] ==gamma_corr[unmodified_mask])
    
    # Within mask, verify probas of states, should be 1 for selected state
    # 0 otherwise
    assert np.all(gamma_corr[state_seq == 0][:,0] == 1)
    assert np.all(gamma_corr[state_seq == 0][:,1:] == 0)
    
    assert np.all(gamma_corr[state_seq == 2][:,2] == 1)
    assert np.all(gamma_corr[state_seq == 2][:,:2] == 0)
    
    assert np.all(gamma_corr[state_seq == 1][:,1] == 1)
    assert np.all(gamma_corr[state_seq == 1][:,[0,2]] == 0)
    
    
    
def test_override_xi_with_None_is_idop():
    """
    When the state sequence is None, we expect to get back exactly original gamma
    """
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask= True,
        
    )
    
    model = Model(config)
    
    n_t = 1000*30
    n_s = 3
    xi = np.random.randn(n_t, n_s*n_s)
    xi_corr = model.override_xi(copy.deepcopy(xi), None)
    
    # We assume EXACT equality
    assert np.all(xi ==xi_corr)
    
    
def test_override_xi_with_NegStateSeq_isNoOP():
    """
    When the state sequence is None, we expect to get back exactly original gamma
    """
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask= True,
        
    )
    
    model = Model(config)
    
    n_t = 1000*30
    n_s = 3
    xi = np.abs(np.random.randn(n_t, n_s*n_s))
    EPS = sys.float_info.epsilon
    xi /= np.expand_dims(np.sum(xi, axis=1), axis=1) + EPS
    states = np.ones(n_t +1)*(-1)
    xi_corr = model.override_xi(copy.deepcopy(xi), states)
    
    # We assume EXACT equality
    assert np.all(xi ==xi_corr)
    
def test_override_xi_is_proper():
    """
    When the state sequence is None, we expect to get back exactly original gamma
    """
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask= True,
        
    )
    
    model = Model(config)
    
    n_t = 1000*30
    n_s = 3
    xi = np.abs(np.random.randn(n_t, n_s*n_s))
    EPS = sys.float_info.epsilon
    xi /= np.expand_dims(np.sum(xi, axis=1), axis=1) + EPS
    states = np.ones(n_t +1, dtype=int)*(-1)
    states[200:250] = 0
    states[400:860] = 2
    states[900:920] = 1
    
    
    xi_corr = model.override_xi(copy.deepcopy(xi), states)
    
    # We assume EXACT equality in segments which do not involve modified states and boundaries
    assert np.all(xi[:199] ==xi_corr[:199])
    assert np.all(xi[252:397] ==xi_corr[252:397])
    assert np.all(xi[862:898] ==xi_corr[862:898])
    assert np.all(xi[922:] ==xi_corr[922:])
    
    # Within segments, the joint distribution should strictly be the self transition of state to itself
    m = xi_corr[201:248]
    v = np.array([[1,0,0],[0,0,0],[0,0,0]]).flatten()
    v_b = np.broadcast_to(v, m.shape)
    assert np.allclose(m,v_b)

def test_gamma_correction():
    """Gamma correction must:
        - Set to 1 the posterior probability at time t of the specified state at time t, to 0 all other posteriors at time t
        - Leave untouched at time t the posterior probabilities if the state is -1
    """
    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    state_seq = np.ones(1000*30)*(-1)
    state_seq[200:250] = 0
    state_seq[400:860] = 2
    state_seq[900:920] = 1
        
    random_means = np.random.randn(3,10)
    
    ##########
    # Generate data
    ##########
        
    
    ####
    # Case 1: default model
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    
    gamma_r = np.abs(np.random.randn(1000*30,3))
    
    assert not np.allclose(gamma_r[200:250,0],1)
    assert not np.allclose(gamma_r[400:860,2],1)
    assert not np.allclose(gamma_r[900:920,1],1)

    gamma_r_cpy = model.override_gamma(copy.deepcopy(gamma_r), state_seq)
    del model
    keras.backend.clear_session()
    assert np.allclose(gamma_r_cpy[200:250,0],1)
    assert np.allclose(gamma_r_cpy[200:250,1:],0)

    assert np.allclose(gamma_r_cpy[400:860,2],1)
    assert np.allclose(gamma_r_cpy[400:860,:2],0)
    assert np.allclose(gamma_r_cpy[900:920,1],1)
    assert np.allclose(gamma_r_cpy[900:920,0],0)
    assert np.allclose(gamma_r_cpy[900:920,2],0)
    
    assert np.allclose(gamma_r_cpy[:200], gamma_r[:200])
    assert np.allclose(gamma_r_cpy[250:400], gamma_r[250:400])
    assert np.allclose(gamma_r_cpy[860:900], gamma_r[860:900])
    assert np.allclose(gamma_r_cpy[920:], gamma_r[920:])
    
def test_xi_correction():
    """Xi correction must:
        - Set to 1 the joint probability (k,j) if s_t = k, s_t+1 = j, 0 to all other values
        - Leave untouched at time t the joint probabilities if any of the two state is -1
    """
    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    state_seq = np.ones(1000*30)*(-1)
    state_seq[200:202] = 0
    state_seq[202:205] = 1
    state_seq[205:220] = 0
    state_seq[220:250] = 2

    state_seq[400:860] = 2
    state_seq[900:920] = 1
        
    random_means = np.random.randn(3,10)
    
    ##########
    # Generate data
    ##########
        
    
    ####
    # Case 1: default model
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    
    xi_r = np.abs(np.random.randn(1000*30 - 1,3*3))
    
    
    assert not np.allclose(xi_r[200:201,0],1) # Case (0x0) => 0 (0*n_states + 0)
    assert not np.allclose(xi_r[200:201,1:],0)

    assert not np.allclose(xi_r[201:202,1],1) # Case (0 x 1) => 1 (0*n_states + 1)
    assert not np.allclose(xi_r[201:202,0],0)
    assert not np.allclose(xi_r[201:202,2:],0)

    assert not np.allclose(xi_r[202:204,4],1) # Case (1 x 1) => 4 (1*n_states + 1)
    assert not np.allclose(xi_r[202:204,:4],0)
    assert not np.allclose(xi_r[202:204,5:],0)
    
    assert not np.allclose(xi_r[204:205,3],1) # Case (1 x 0) => 3 (1*n_states + 0)
    assert not np.allclose(xi_r[204:205,:2],0)
    assert not np.allclose(xi_r[204:205,4:],0)


    xi_r_copy = model.override_xi(copy.deepcopy(xi_r), state_seq)
    del model
    keras.backend.clear_session()
    print(xi_r_copy[200:202])
    assert np.allclose(xi_r_copy[200:201,0],1) # Case (0x0) => 0 (0*n_states + 0)
    assert np.allclose(xi_r_copy[200:201,1:],0)

    assert np.allclose(xi_r_copy[201:202,1],1) # Case (0 x 1) => 1 (0*n_states + 1)
    assert np.allclose(xi_r_copy[201:202,0],0)
    assert np.allclose(xi_r_copy[201:202,2:],0)

    assert np.allclose(xi_r_copy[202:204,4],1) # Case (1 x 1) => 4 (1*n_states + 1)
    assert np.allclose(xi_r_copy[202:204,:4],0)
    assert np.allclose(xi_r_copy[202:204,5:],0)
    
    assert np.allclose(xi_r_copy[204:205,3],1) # Case (1 x 0) => 3 (1*n_states + 0)
    assert np.allclose(xi_r_copy[204:205,:2],0)
    assert np.allclose(xi_r_copy[204:205,4:],0)
    
    assert np.allclose(xi_r_copy[205:219,0],1) # Case (0 x 0) => 0 (0*n_states + 0)
    assert np.allclose(xi_r_copy[205:219,1:],0)
    

    assert np.allclose(xi_r_copy[219:220,2],1) # Case (0 x 2) => 2 (0*n_states + 2)
    assert np.allclose(xi_r_copy[219:220,:2],0) # Case (0 x 2) => 2 (0*n_states + 2)
    assert np.allclose(xi_r_copy[219:220,3:],0) # Case (0 x 2) => 2 (0*n_states + 2)


    # Untouched parts should NOT have changed!
    assert np.allclose(xi_r_copy[:200], xi_r[:200])
    assert np.allclose(xi_r_copy[250:400], xi_r[250:400])
    assert np.allclose(xi_r_copy[860:900], xi_r[860:900])
    assert np.allclose(xi_r_copy[920:], xi_r[920:])

def test_get_alpha_with_options_but_None_gives_base_val():
    """
    Calling get_alpha on default model and use_mask model with None mask or semi_supervised model with None state seq
    must yield exactly the same results.
    """    
    ########
    # Setup model parameters
    ########
    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    state_seq = np.ones(10000)*(-1)
    state_seq[400:450]=1
    state_seq[450:460] = 2
        
    random_means = np.random.randn(3,10)
    
    ##########
    # Generate data
    ##########
    
    data_ = np.random.randn(10000, 10)
    data = Data([data_],time_axis_first=True, sampling_frequency=250.0)
    mask = np.ones(10000,dtype=bool)
    mask[400:700] = 0
    mask[1350:1781] = 0
    
    keras.utils.set_random_seed(812)

    
    ####
    # Case 1: default model
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    h = model.fit(data)
    alpha_base = model.get_alpha(data)
    del model
    keras.backend.clear_session()
    
    
    ####
    # Case 2: masked model
    ####
    keras.utils.set_random_seed(812)

    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        use_mask=True,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    h = model.fit(data)
    alpha_mask = model.get_alpha(data)
    del model
    keras.backend.clear_session()
    
    
    ####
    # Case 3: state seq model
    ####
    keras.utils.set_random_seed(812)

    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        semi_supervised=True,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    h = model.fit(data)
    alpha_state_seq = model.get_alpha(data)
    del model
    keras.backend.clear_session()
    
    ##################
    # Case 4: both
    ##################
    keras.utils.set_random_seed(812)

    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        semi_supervised=True,        
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    #assert isinstance(data,list)
    h = model.fit(data)
    alpha_state_seq_mask = model.get_alpha(data)
    del model
    keras.backend.clear_session()
    
    #state_seq[400:450]=1
    #state_seq[450:460] = 2
    
    #assert alpha_state_seq_mask.shape == alpha_base.shape
    #assert np.allclose(alpha_state_seq_mask[400:450,1],1) 
    #assert alpha_state_seq_
    assert np.allclose(alpha_base, alpha_state_seq)
    assert np.allclose(alpha_base, alpha_state_seq_mask)
    assert np.allclose(alpha_base, alpha_mask)
    #assert True == False
    
def test_mask_influences_tc():
    """
    Masking with state censoring should affect
    posterior state probabilities.
    In particular, we should expect that the probabilities
    which are hard set should be 1.
    """
    ########
    # Setup model parameters
    ########
    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    state_seq = np.ones(10000)*(-1)
    state_seq[400:450]=1
    state_seq[450:460] = 2
        
    random_means = np.random.randn(3,10)
    
    ##########
    # Generate data
    ##########
    
    data_ = np.random.randn(10000, 10)
    data = Data([data_],time_axis_first=True, sampling_frequency=250.0)
    mask = np.ones(10000,dtype=bool)
    mask[400:700] = 0
    mask[1350:1781] = 0
    
    keras.utils.set_random_seed(812)

    
    ####
    # Case 1: default model
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    h = model.fit(data)
    alpha_base = model.get_alpha(data)
    del model
    keras.backend.clear_session()
    
    
    ####
    # Case 2: state seq model
    ####
    keras.utils.set_random_seed(812)

    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        semi_supervised=True,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    h = model.fit(data, forced_states=state_seq)
    alpha_state_seq = model.get_alpha(data, forced_states=state_seq)
    del model
    keras.backend.clear_session()
    
    ##################
    # Case 3: both
    ##################
    keras.utils.set_random_seed(812)

    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        semi_supervised=True,        
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    #assert isinstance(data,list)
    h = model.fit(data, forced_states=state_seq)
    alpha_state_seq_mask = model.get_alpha(data, forced_states=state_seq)
    del model
    keras.backend.clear_session()
    
    #assert np.allclose(alpha_state_seq_mask[400:450,1],1) 
    #assert np.allclose(alpha_state_seq[400:450,1],1) 
    #assert np.allclose(alpha_state_seq, alpha_state_seq_mask)

    #assert not np.allclose(alpha_base, alpha_state_seq_mask)
    
def test_likelihood_mask():
    """
    Likelihood masking specific points is intended to
    remove said points from the learning.
    As such, learnt parameters should differ, specifically
    in the case of outlying points.
    """
    assert True == False
    
def test_seed():
    """
    Calling get_alpha on default model and use_mask model with None mask or semi_supervised model with None state seq
    must yield exactly the same results.
    """    
    ########
    # Setup model parameters
    ########
    random_covs = np.zeros((3,10,10))
    for i in range(3):
        a = np.random.randn(10,10)
        random_covs[i] = a.T @ a
    random_T = np.abs(np.random.randn(3,3))
    for i in range(3):
        random_T[i] /= random_T[i].sum()
        
    state_seq = np.ones(10000)*(-1)
    state_seq[400:450]=1
    state_seq[450:460] = 2
        
    random_means = np.random.randn(3,10)
    
    ##########
    # Generate data
    ##########
    
    data_ = np.random.randn(10000, 10)
    data = Data([data_],time_axis_first=True, sampling_frequency=250.0)
    mask = np.ones(10000,dtype=bool)
    mask[400:700] = 0
    mask[1350:1781] = 0
    
    keras.utils.set_random_seed(812)
    ####
    # Case 1: default model
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    h = model.fit(data)
    alpha_base = model.get_alpha(data)
    del model
    keras.backend.clear_session()
    
    keras.utils.set_random_seed(812)
     ####
    # Case 1: default model
    ####
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=True,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=5,
        multi_gpu = False,
        initial_means=random_means,
        initial_covariances=random_covs,
        initial_trans_prob=random_T,
        state_probs_t0 = np.ones(3)/3
        
    )
    
    model = Model(config)
    h = model.fit(data)
    alpha_two = model.get_alpha(data)
    del model
    keras.backend.clear_session()
    
    
    assert np.allclose(alpha_base, alpha_two)
    
def test_hmm_kappa_increases_diag():
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
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=10,
        multi_gpu = False,
        use_mask= False,
        kappa=0.9
    )
    
    model = Model(config)


    data = Data([np.random.randn(10000, 10)],time_axis_first=True, sampling_frequency=250.0)
    
    model.fit(data)
    
    trans_prob_1 = model.trans_prob
    
    
    config = Config(
        n_states=3,
        n_channels=10,
        sequence_length=1000,
        learn_means=False,
        learn_covariances=True,
        batch_size=30,
        learning_rate=0.01,
        n_epochs=10,
        multi_gpu = False,
        use_mask= False,
    )
    
    model = Model(config)
    model.fit(data)
    
    
    trans_prob_2 = model.trans_prob


    assert np.all(np.diag(trans_prob_1) > np.diag(trans_prob_2))
    
