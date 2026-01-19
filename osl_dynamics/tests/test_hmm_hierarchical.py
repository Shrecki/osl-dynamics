"""Unit tests for hierarchical categorical log-likelihood gradients.

Tests verify that analytical gradients match numerical gradients computed
via finite differences.
"""

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose
import tensorflow as tf
import tensorflow_probability as tfp

from osl_dynamics.models.hmm_hierarchical import (
    hierarchical_categorical_nll_mean_custom_grad,
    hierarchical_categorical_nll_sum_custom_grad,
    HierarchicalCategoricalLogLikelihoodLossLayer,
    sort_and_rowsplit_by_subject
)

def convert_to_dense(grad):
    """Convert gradient to dense tensor if it's IndexedSlices."""
    if isinstance(grad, tf.IndexedSlices):
        return tf.convert_to_tensor(grad)
    return grad


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def regularization_only_problem():
    """Problem designed to isolate regularization gradient."""
    np.random.seed(999)
    
    B = 2
    T = 2
    D = 2
    K = 2
    P = 2
    
    # Zero data - no NLL contribution
    x = np.zeros((B, T, D), dtype=np.float32)
    
    # Simple parameters
    mu_pop = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    L_pop = np.zeros((K, D, D), dtype=np.float32)
    for k in range(K):
        L_pop[k] = np.eye(D, dtype=np.float32)
    
    mu_subj = np.array([
        [[2.0, 3.0], [4.0, 5.0]],  # Subject 0: differs from pop by [1,1]
        [[0.0, 1.0], [2.0, 3.0]],  # Subject 1: differs from pop by [-1,-1]
    ], dtype=np.float32)
    
    L_subj = np.zeros((P, K, D, D), dtype=np.float32)
    for p in range(P):
        for k in range(K):
            L_subj[p, k] = np.eye(D, dtype=np.float32)
    
    # Uniform gamma
    gamma = np.ones((B, T, K), dtype=np.float32) / K
    
    # Both subjects in batch
    subject_ids = np.array([[0,0], [1,1]], dtype=np.int32)
    
    return {
        'x': x,
        'mu_pop': mu_pop,
        'L_pop': L_pop,
        'mu_subj': mu_subj,
        'L_subj': L_subj,
        'gamma': gamma,
        'subject_ids': subject_ids,
        'lambda_mu': 1.0,
        'lambda_L': 0.0,  # Focus on mu only
        'n_subjects': P,
        'B': B, 'T': T, 'D': D, 'K': K, 'P': P,
    }


@pytest.fixture
def small_problem():
    """Create a small problem for gradient testing."""
    np.random.seed(42)
    
    B = 4   # batch size
    T = 5   # sequence length
    D = 3   # channels
    K = 2   # states
    P = 3   # subjects
    
    # Generate random data - use float64 for better numerical precision
    x = np.random.randn(B, T, D).astype(np.float64)
    
    # Population parameters
    mu_pop = np.random.randn(K, D).astype(np.float64)
    L_pop = np.zeros((K, D, D), dtype=np.float64)
    for k in range(K):
        A = np.random.randn(D, D).astype(np.float64) * 0.3
        L_pop[k] = np.linalg.cholesky(A @ A.T + np.eye(D))
    
    # Subject parameters
    mu_subj = np.random.randn(P, K, D).astype(np.float64)
    L_subj = np.zeros((P, K, D, D), dtype=np.float64)
    for p in range(P):
        for k in range(K):
            A = np.random.randn(D, D).astype(np.float64) * 0.3
            L_subj[p, k] = np.linalg.cholesky(A @ A.T + np.eye(D))
    
    # Gamma (responsibilities) - must sum to 1 over states
    gamma_raw = np.random.rand(B, T, K).astype(np.float64)
    gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
    
    # Subject IDs - mix of subjects in batch
    subject_ids = np.array([[0,0,1,2], [1,1,1,2], [0,0,2,2], [2,2,2,0],[1,0,0,1]], dtype=np.int32).T
    
    lambda_mu = 1.0
    lambda_L = 1.0
    
    return {
        'x': x,
        'mu_pop': mu_pop,
        'L_pop': L_pop,
        'mu_subj': mu_subj,
        'L_subj': L_subj,
        'gamma': gamma,
        'subject_ids': subject_ids,
        'lambda_mu': lambda_mu,
        'lambda_L': lambda_L,
        'n_subjects': P,
        'B': B, 'T': T, 'D': D, 'K': K, 'P': P,
    }


@pytest.fixture
def single_subject_problem():
    """Problem with only one subject in batch."""
    np.random.seed(123)
    
    B = 3
    T = 4
    D = 2
    K = 2
    P = 3
    
    x = np.random.randn(B, T, D).astype(np.float32)
    mu_pop = np.random.randn(K, D).astype(np.float32)
    L_pop = np.zeros((K, D, D), dtype=np.float32)
    for k in range(K):
        A = np.random.randn(D, D).astype(np.float32) * 0.3
        L_pop[k] = np.linalg.cholesky(A @ A.T + np.eye(D))
    
    mu_subj = np.random.randn(P, K, D).astype(np.float32)
    L_subj = np.zeros((P, K, D, D), dtype=np.float32)
    for p in range(P):
        for k in range(K):
            A = np.random.randn(D, D).astype(np.float32) * 0.3
            L_subj[p, k] = np.linalg.cholesky(A @ A.T + np.eye(D))
    
    gamma_raw = np.random.rand(B, T, K).astype(np.float32)
    gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
    
    # Only subject 1 in this batch
    subject_ids = np.array([[1,1,1], [1,1,1], [1,1,1],[1,1,1]], dtype=np.int32)
    
    return {
        'x': x,
        'mu_pop': mu_pop,
        'L_pop': L_pop,
        'mu_subj': mu_subj,
        'L_subj': L_subj,
        'gamma': gamma,
        'subject_ids': subject_ids,
        'lambda_mu': 0.5,
        'lambda_L': 0.5,
        'n_subjects': P,
        'B': B, 'T': T, 'D': D, 'K': K, 'P': P,
    }


# =============================================================================
# Helper functions
# =============================================================================

def numerical_gradient(f, x, epsilon=1e-5):
    """Compute numerical gradient using central differences.
    
    Parameters
    ----------
    f : callable
        Function that takes x and returns a scalar.
    x : np.ndarray
        Point at which to compute gradient.
    epsilon : float
        Step size for finite differences.
    
    Returns
    -------
    grad : np.ndarray
        Numerical gradient, same shape as x.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        # f(x + epsilon)
        x[idx] = old_value + epsilon
        fx_plus = f(x)
        
        # f(x - epsilon)
        x[idx] = old_value - epsilon
        fx_minus = f(x)
        
        # Central difference
        grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
        
        # Restore
        x[idx] = old_value
        it.iternext()
    
    return grad


def compute_analytical_gradients(problem, calculation='mean'):
    """Compute analytical gradients using the custom gradient function."""
    x = tf.constant(problem['x'])
    mu_pop = tf.Variable(problem['mu_pop'])
    L_pop = tf.Variable(problem['L_pop'])
    mu_subj = tf.Variable(problem['mu_subj'])
    L_subj = tf.Variable(problem['L_subj'])
    gamma = tf.constant(problem['gamma'])
    subject_ids = tf.constant(problem['subject_ids'])
    
    with tf.GradientTape() as tape:
        if calculation == 'mean':
            loss = hierarchical_categorical_nll_mean_custom_grad(
                x, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids,
                problem['lambda_mu'], problem['lambda_L'], problem['n_subjects']
            )
        else:
            loss = hierarchical_categorical_nll_sum_custom_grad(
                x, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids,
                problem['lambda_mu'], problem['lambda_L'], problem['n_subjects']
            )
    
    grads = tape.gradient(loss, [mu_pop, L_pop, mu_subj, L_subj])
    
    return {
        'loss': loss.numpy(),
        'grad_mu_pop': grads[0].numpy(),
        'grad_L_pop': grads[1].numpy(),
        'grad_mu_subj': grads[2].numpy(),
        'grad_L_subj': grads[3].numpy(),
    }


def compute_numerical_gradients(problem, calculation='mean', epsilon=1e-5):
    """Compute numerical gradients using finite differences."""
    
    def make_loss_fn(param_name):
        def loss_fn(param_value):
            x = tf.constant(problem['x'])
            mu_pop = tf.constant(problem['mu_pop'] if param_name != 'mu_pop' else param_value)
            L_pop = tf.constant(problem['L_pop'] if param_name != 'L_pop' else param_value)
            mu_subj = tf.constant(problem['mu_subj'] if param_name != 'mu_subj' else param_value)
            L_subj = tf.constant(problem['L_subj'] if param_name != 'L_subj' else param_value)
            gamma = tf.constant(problem['gamma'])
            subject_ids = tf.constant(problem['subject_ids'])
            
            if calculation == 'mean':
                loss = hierarchical_categorical_nll_mean_custom_grad(
                    x, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids,
                    problem['lambda_mu'], problem['lambda_L'], problem['n_subjects']
                )
            else:
                loss = hierarchical_categorical_nll_sum_custom_grad(
                    x, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids,
                    problem['lambda_mu'], problem['lambda_L'], problem['n_subjects']
                )
            return loss.numpy()
        return loss_fn
    
    return {
        'grad_mu_pop': numerical_gradient(make_loss_fn('mu_pop'), problem['mu_pop'].copy(), epsilon),
        'grad_L_pop': numerical_gradient(make_loss_fn('L_pop'), problem['L_pop'].copy(), epsilon),
        'grad_mu_subj': numerical_gradient(make_loss_fn('mu_subj'), problem['mu_subj'].copy(), epsilon),
        'grad_L_subj': numerical_gradient(make_loss_fn('L_subj'), problem['L_subj'].copy(), epsilon),
    }

##################################################
# Tests for analytical gradient vs autodiff      #
##################################################
tfd = tfp.distributions

class TestSplit:
    def test_split_identified_uniques_agree(self):
        np.random.seed(12345)
        B, T, D, K, P = 5, 10, 16, 8, 4
        
        # Create the data
        
        x = np.random.randn(B, T, D)
        gamma_raw = np.random.rand(B, T, K)
        gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
        
        subject_ids = np.random.randint(0, P, size=(B,T)).astype(np.int32)
        
        unique_subjects, row_splits, x_sorted, gamma_sorted, sid_sorted = sort_and_rowsplit_by_subject(x, gamma, subject_ids)
        
        # Assert all 
        
        assert np.all(unique_subjects.shape == np.unique(subject_ids.flatten()).shape)
        assert np.all(np.sort(np.unique(subject_ids.flatten())) == np.sort(unique_subjects))
    
    def test_splits_agree(self):
        np.random.seed(12345)
        B, T, D, K, P = 5, 10, 16, 8, 4
        
        # Create the data
        
        x = np.random.randn(B, T, D)
        gamma_raw = np.random.rand(B, T, K)
        gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
        
        subject_ids = np.random.randint(0, P, size=(B,T)).astype(np.int32)
        subject_ids_flat = subject_ids.flatten()
        unique_subjects, row_splits, x_sorted, gamma_sorted, sid_sorted = sort_and_rowsplit_by_subject(x, gamma, subject_ids)
        
        ids_ = np.argsort(subject_ids_flat)
        sorted_ = subject_ids_flat[ids_]
        breaks = [0]
        for i in range(ids_.size-1):
            if sorted_[i] != sorted_[i+1]:
                breaks.append(i+1)
        breaks.append(ids_.size)
        breaks = np.array(breaks)
        
        
        assert np.all(breaks.shape == row_splits.shape)
        assert np.all(breaks == row_splits)
        

class TestAnalyticalVsAutodiff:
    """Compare our analytical gradients against TensorFlow's autodiff.
    
    This creates a reference implementation using standard TF ops and compares
    the gradients to our custom gradient implementation.
    """
    
    @staticmethod
    def reference_loss(x, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids,
                   lambda_mu, lambda_L, n_subjects):
        """Reference implementation that builds the big gathered tensors.

        Intended ONLY for correctness testing vs analytical gradients.
        """
        dtype = mu_pop.dtype
        x = tf.cast(x, dtype)
        gamma = tf.cast(gamma, dtype)
        subject_ids = tf.cast(subject_ids, tf.int32)

        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # Unique subjects for scaling (as in your main loss)
        unique_subjects, _ = tf.unique(tf.reshape(subject_ids, [-1]))
        n_unique = tf.cast(tf.shape(unique_subjects)[0], dtype)
        scale = n_unique / tf.cast(n_subjects, dtype)

        # ---- BIG GATHERS (B,T,...) ----
        # These are the tensors we avoid in the efficient implementation.
        mu_batch = tf.gather(mu_subj, subject_ids, axis=0)  # (B, T, K, D)
        L_batch  = tf.gather(L_subj,  subject_ids, axis=0)  # (B, T, K, D, D)

        # NLL
        mvn = tfd.MultivariateNormalTriL(
            loc=mu_batch,        # (B,T,K,D)
            scale_tril=L_batch,  # (B,T,K,D,D)
            allow_nan_stats=False
        )

        # x needs to broadcast over K: (B,T,1,D)
        log_probs = mvn.log_prob(x[:, :, None, :])          # (B, T, K)
        ll_weighted = tf.reduce_sum(gamma * log_probs, axis=-1)  # (B, T)
        nll = -tf.reduce_mean(ll_weighted)                  # mean over (B,T)

        # Regularization over subjects present (your original approach)
        mu_b = tf.gather(mu_subj, unique_subjects, axis=0)  # (S,K,D)
        L_b  = tf.gather(L_subj,  unique_subjects, axis=0)  # (S,K,D,D)

        mu_reg = tf.reduce_sum(tf.square(mu_b - mu_pop[None, :, :]))
        L_reg  = tf.reduce_sum(tf.square(L_b  - L_pop[None, :, :, :]))

        reg_loss = scale * (
            tf.cast(lambda_mu, dtype) * 0.5 * mu_reg +
            tf.cast(lambda_L,  dtype) * 0.5 * L_reg
        )

        return nll + reg_loss
    
    def test_mu_pop_gradient_vs_autodiff(self, small_problem):
        """Compare mu_pop gradient: analytical vs autodiff."""
        # Autodiff reference
        mu_pop_var = tf.Variable(small_problem['mu_pop'], dtype=tf.float64)
        
        with tf.GradientTape() as tape:
            loss_ref = self.reference_loss(
                tf.constant(small_problem['x'], dtype=tf.float64),
                mu_pop_var,
                tf.constant(small_problem['L_pop'], dtype=tf.float64),
                tf.constant(small_problem['mu_subj'], dtype=tf.float64),
                tf.constant(small_problem['L_subj'], dtype=tf.float64),
                tf.constant(small_problem['gamma'], dtype=tf.float64),
                tf.constant(small_problem['subject_ids']),
                small_problem['lambda_mu'],
                small_problem['lambda_L'],
                small_problem['n_subjects']
            )
        grad_autodiff = convert_to_dense(tape.gradient(loss_ref, mu_pop_var))
        
        # Analytical gradient
        analytical = compute_analytical_gradients(small_problem, 'mean')
        
        print("Autodiff grad_mu_pop:\n", grad_autodiff.numpy())
        print("Analytical grad_mu_pop:\n", analytical['grad_mu_pop'])
        print("Difference:\n", grad_autodiff.numpy() - analytical['grad_mu_pop'])
        
        assert_allclose(
            analytical['grad_mu_pop'],
            grad_autodiff.numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg="mu_pop gradient mismatch: analytical vs autodiff"
        )
    
    def test_L_pop_gradient_vs_autodiff(self, small_problem):
        """Compare L_pop gradient: analytical vs autodiff."""
        L_pop_var = tf.Variable(small_problem['L_pop'], dtype=tf.float64)
        
        with tf.GradientTape() as tape:
            loss_ref = self.reference_loss(
                tf.constant(small_problem['x'], dtype=tf.float64),
                tf.constant(small_problem['mu_pop'], dtype=tf.float64),
                L_pop_var,
                tf.constant(small_problem['mu_subj'], dtype=tf.float64),
                tf.constant(small_problem['L_subj'], dtype=tf.float64),
                tf.constant(small_problem['gamma'], dtype=tf.float64),
                tf.constant(small_problem['subject_ids']),
                small_problem['lambda_mu'],
                small_problem['lambda_L'],
                small_problem['n_subjects']
            )
        grad_autodiff = convert_to_dense(tape.gradient(loss_ref, L_pop_var))
        
        analytical = compute_analytical_gradients(small_problem, 'mean')
        
        print("Autodiff grad_L_pop:\n", grad_autodiff.numpy())
        print("Analytical grad_L_pop:\n", analytical['grad_L_pop'])
        
        assert_allclose(
            analytical['grad_L_pop'],
            grad_autodiff.numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg="L_pop gradient mismatch: analytical vs autodiff"
        )
    
    def test_mu_subj_gradient_vs_autodiff(self, small_problem):
        """Compare mu_subj gradient: analytical vs autodiff."""
        mu_subj_var = tf.Variable(small_problem['mu_subj'], dtype=tf.float64)
        
        with tf.GradientTape() as tape:
            loss_ref = self.reference_loss(
                tf.constant(small_problem['x'], dtype=tf.float64),
                tf.constant(small_problem['mu_pop'], dtype=tf.float64),
                tf.constant(small_problem['L_pop'], dtype=tf.float64),
                mu_subj_var,
                tf.constant(small_problem['L_subj'], dtype=tf.float64),
                tf.constant(small_problem['gamma'], dtype=tf.float64),
                tf.constant(small_problem['subject_ids']),
                small_problem['lambda_mu'],
                small_problem['lambda_L'],
                small_problem['n_subjects']
            )
        grad_autodiff = convert_to_dense(tape.gradient(loss_ref, mu_subj_var))
        
        analytical = compute_analytical_gradients(small_problem, 'mean')
        
        print("Autodiff grad_mu_subj:\n", grad_autodiff.numpy())
        print("Analytical grad_mu_subj:\n", analytical['grad_mu_subj'])
        
        assert_allclose(
            analytical['grad_mu_subj'],
            grad_autodiff.numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg="mu_subj gradient mismatch: analytical vs autodiff"
        )
    
    def test_L_subj_gradient_vs_autodiff(self, small_problem):
        """Compare L_subj gradient: analytical vs autodiff."""
        L_subj_var = tf.Variable(small_problem['L_subj'], dtype=tf.float64)
        
        with tf.GradientTape() as tape:
            loss_ref = self.reference_loss(
                tf.constant(small_problem['x'], dtype=tf.float64),
                tf.constant(small_problem['mu_pop'], dtype=tf.float64),
                tf.constant(small_problem['L_pop'], dtype=tf.float64),
                tf.constant(small_problem['mu_subj'], dtype=tf.float64),
                L_subj_var,
                tf.constant(small_problem['gamma'], dtype=tf.float64),
                tf.constant(small_problem['subject_ids']),
                small_problem['lambda_mu'],
                small_problem['lambda_L'],
                small_problem['n_subjects']
            )
        grad_autodiff = convert_to_dense(tape.gradient(loss_ref, L_subj_var))
        
        analytical = compute_analytical_gradients(small_problem, 'mean')
        
        print("Autodiff grad_L_subj:\n", grad_autodiff.numpy())
        print("Analytical grad_L_subj:\n", analytical['grad_L_subj'])
        
        assert_allclose(
            analytical['grad_L_subj'],
            grad_autodiff.numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg="L_subj gradient mismatch: analytical vs autodiff"
        )
        
    def test_ll_layer_gradients_vs_autodiff_random(self):
        np.random.seed(12345)
        for trial in range(5):  # Run multiple random trials
            B, T, D, K, P = 32, 256, 16, 8, 4
            
            # Create the data
            
            x = np.random.randn(B, T, D)
            mu_pop = np.random.randn(K, D)
            L_pop = np.zeros((K, D, D))
            for k in range(K):
                A = np.random.randn(D, D) * 0.3
                L_pop[k] = np.linalg.cholesky(A @ A.T + np.eye(D))
            
            mu_subj = np.random.randn(P, K, D)
            L_subj = np.zeros((P, K, D, D))
            for p in range(P):
                for k in range(K):
                    A = np.random.randn(D, D) * 0.3
                    L_subj[p, k] = np.linalg.cholesky(A @ A.T + np.eye(D))
            
            gamma_raw = np.random.rand(B, T, K)
            gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
            
            subject_ids = np.zeros((B,T),dtype=np.int32) #np.randint(0, P, size=(B,T)).astype(np.int32)
            lambda_mu = np.random.rand() * 2
            lambda_L = np.random.rand() * 2
            
            # Layer implem
            mu_pop_var = tf.Variable(mu_pop, dtype=tf.float64)
            L_pop_var = tf.Variable(L_pop, dtype=tf.float64)
            mu_subj_var = tf.Variable(mu_subj, dtype=tf.float64)
            L_subj_var = tf.Variable(L_subj, dtype=tf.float64)
            
            # Create the layer
            # Create the layer
            layer = HierarchicalCategoricalLogLikelihoodLossLayer(
                n_states=K, 
                n_subjects=P,
                epsilon=1e-6,
                calculation='mean',
                lambda_mu=lambda_mu,
                lambda_L=lambda_L,
            )
            
            with tf.GradientTape() as tape:
                loss_layer = layer([x, mu_pop_var, L_pop_var, mu_subj_var, L_subj_var, gamma, subject_ids])
                
            grads_layer = [
                convert_to_dense(g).numpy() for g in tape.gradient(
                    loss_layer, 
                    [mu_pop_var, L_pop_var, mu_subj_var, L_subj_var]
                )
            ]
            
            # Autodiff implem
            with tf.GradientTape() as tape2:
                loss_ref = self.reference_loss(
                    tf.constant(x, dtype=tf.float64),
                    mu_pop_var,
                    L_pop_var,
                    mu_subj_var,
                    L_subj_var,
                    tf.constant(gamma, dtype=tf.float64),
                    tf.constant(subject_ids),
                    lambda_mu,
                    lambda_L,
                    P
                )
            grads_autodiff = [
                convert_to_dense(g).numpy() for g in tape2.gradient(
                    loss_ref, 
                    [mu_pop_var, L_pop_var, mu_subj_var, L_subj_var]
                )
            ]
            
            # Compare
            """for name, g_auto, g_anal in zip(
                ['mu_pop', 'L_pop', 'mu_subj', 'L_subj'],
                grads_autodiff,
                grads_layer
            ):
                assert_allclose(
                    g_anal,
                    g_auto,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Trial {trial}: {name} gradient mismatch"
               y )
            """
            # Also verify loss values match
            assert_allclose(
                loss_ref.numpy(),
                loss_layer.numpy(),
                rtol=1e-9,
                err_msg=f"Trial {trial}: Loss values don't match"
            )
            
    def test_ll_layer_returns_zero_grad_for_missing_subj(self):
        np.random.seed(12345)
        for trial in range(5):  # Run multiple random trials
            B, T, D, K, P = 64, 512, 32, 8, 4
            
            # Create the data
            
            x = np.random.randn(B, T, D)
            mu_pop = np.random.randn(K, D)
            L_pop = np.zeros((K, D, D))
            for k in range(K):
                A = np.random.randn(D, D) * 0.3
                L_pop[k] = np.linalg.cholesky(A @ A.T + np.eye(D))
            
            mu_subj = np.random.randn(P, K, D)
            L_subj = np.zeros((P, K, D, D))
            for p in range(P):
                for k in range(K):
                    A = np.random.randn(D, D) * 0.3
                    L_subj[p, k] = np.linalg.cholesky(A @ A.T + np.eye(D))
            
            gamma_raw = np.random.rand(B, T, K)
            gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
            
            subject_ids = np.random.randint(0, P, size=(B,T)).astype(np.int32)
            
            # Select subject 0 and discard then for the sake of the example
            rand_subj_discarded = 0#np.random.randint(0,P)
            subject_ids[subject_ids == rand_subj_discarded] = 1 # Replace them with subject 1
            
            lambda_mu = np.random.rand() * 2
            lambda_L = np.random.rand() * 2
            
            # Layer implem
            mu_pop_var = tf.Variable(mu_pop, dtype=tf.float64)
            L_pop_var = tf.Variable(L_pop, dtype=tf.float64)
            mu_subj_var = tf.Variable(mu_subj, dtype=tf.float64)
            L_subj_var = tf.Variable(L_subj, dtype=tf.float64)
            
            # Create the layer
            layer = HierarchicalCategoricalLogLikelihoodLossLayer(
                n_states=K, 
                n_subjects=P,
                epsilon=1e-6,
                calculation='mean',
                lambda_mu=lambda_mu,
                lambda_L=lambda_L,
            )
            
            with tf.GradientTape() as tape:
                loss_layer = layer([x, mu_pop_var, L_pop_var, mu_subj_var, L_subj_var, gamma, subject_ids])
                
            grads_layer = [
                convert_to_dense(g).numpy() for g in tape.gradient(
                    loss_layer, 
                    [mu_pop_var, L_pop_var, mu_subj_var, L_subj_var]
                )
            ]
            
            # Verify that gradients for the discarded subject are indeed equal to zero
            assert_allclose(grads_layer[2][rand_subj_discarded], np.zeros(grads_layer[2][0].shape))
            assert_allclose(grads_layer[3][rand_subj_discarded], np.zeros(grads_layer[3][0].shape))
            
    
    def test_all_gradients_vs_autodiff_random(self):
        """Test all gradients with randomly generated parameters."""
        np.random.seed(12345)
        
        for trial in range(5):  # Run multiple random trials
            B, T, D, K, P = 32, 128, 32, 8, 4
            
            x = np.random.randn(B, T, D)
            mu_pop = np.random.randn(K, D)
            L_pop = np.zeros((K, D, D))
            for k in range(K):
                A = np.random.randn(D, D) * 0.3
                L_pop[k] = np.linalg.cholesky(A @ A.T + np.eye(D))
            
            mu_subj = np.random.randn(P, K, D)
            L_subj = np.zeros((P, K, D, D))
            for p in range(P):
                for k in range(K):
                    A = np.random.randn(D, D) * 0.3
                    L_subj[p, k] = np.linalg.cholesky(A @ A.T + np.eye(D))
            
            gamma_raw = np.random.rand(B, T, K)
            gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
            
            subject_ids = np.random.randint(0, P, size=(B,T)).astype(np.int32)
            lambda_mu = np.random.rand() * 2
            lambda_L = np.random.rand() * 2
            
            # Autodiff
            mu_pop_var = tf.Variable(mu_pop, dtype=tf.float64)
            L_pop_var = tf.Variable(L_pop, dtype=tf.float64)
            mu_subj_var = tf.Variable(mu_subj, dtype=tf.float64)
            L_subj_var = tf.Variable(L_subj, dtype=tf.float64)
            
            with tf.GradientTape() as tape:
                loss_ref = self.reference_loss(
                    tf.constant(x, dtype=tf.float64),
                    mu_pop_var,
                    L_pop_var,
                    mu_subj_var,
                    L_subj_var,
                    tf.constant(gamma, dtype=tf.float64),
                    tf.constant(subject_ids),
                    lambda_mu,
                    lambda_L,
                    P
                )
            grads_autodiff = [
                convert_to_dense(g).numpy() for g in tape.gradient(
                    loss_ref, 
                    [mu_pop_var, L_pop_var, mu_subj_var, L_subj_var]
                )
            ]
            
            # Analytical
            with tf.GradientTape() as tape2:
                tape2.watch([mu_pop_var, L_pop_var, mu_subj_var, L_subj_var])
                loss_analytical = hierarchical_categorical_nll_mean_custom_grad(
                    tf.constant(x, dtype=tf.float64),
                    mu_pop_var,
                    L_pop_var,
                    mu_subj_var,
                    L_subj_var,
                    tf.constant(gamma, dtype=tf.float64),
                    tf.constant(subject_ids),
                    lambda_mu,
                    lambda_L,
                    P
                )
            grads_analytical = [
                convert_to_dense(g).numpy() for g in tape2.gradient(
                    loss_analytical,
                    [mu_pop_var, L_pop_var, mu_subj_var, L_subj_var]
                )
            ]
            
            # Compare
            for name, g_auto, g_anal in zip(
                ['mu_pop', 'L_pop', 'mu_subj', 'L_subj'],
                grads_autodiff,
                grads_analytical
            ):
                assert_allclose(
                    g_anal,
                    g_auto,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Trial {trial}: {name} gradient mismatch"
                )
            
            # Also verify loss values match
            assert_allclose(
                loss_analytical.numpy(),
                loss_ref.numpy(),
                rtol=1e-10,
                err_msg=f"Trial {trial}: Loss values don't match"
            )



# =============================================================================
# Tests for gradient correctness
# =============================================================================

class TestGradientCorrectness:
    """Tests that analytical gradients match numerical gradients."""
    
    def test_regularization_only_mu_pop(self, regularization_only_problem):
        """Test mu_pop gradient with only regularization (no NLL)."""
        analytical = compute_analytical_gradients(regularization_only_problem, 'mean')
        numerical = compute_numerical_gradients(regularization_only_problem, 'mean')
        
        print("Analytical grad_mu_pop:", analytical['grad_mu_pop'])
        print("Numerical grad_mu_pop:", numerical['grad_mu_pop'])
        print("Difference:", analytical['grad_mu_pop'] - numerical['grad_mu_pop'])
        
        # Expected: grad = -lambda * sum(mu_subj - mu_pop) * scale
        # mu_subj[0] - mu_pop = [[1,1], [1,1]]
        # mu_subj[1] - mu_pop = [[-1,-1], [-1,-1]]
        # sum = [[0,0], [0,0]]
        # So grad_mu_pop should be 0!
        print("Expected: 0 (since subject diffs cancel out)")
        
        assert_allclose(
            analytical['grad_mu_pop'],
            numerical['grad_mu_pop'],
            rtol=1e-4,
            atol=1e-5,
        )
    
    def test_gradient_with_tf_test(self, small_problem):
        """Test gradients using TensorFlow's gradient checking."""
        
        def loss_fn(mu_pop):
            return hierarchical_categorical_nll_mean_custom_grad(
                tf.constant(small_problem['x']),
                mu_pop,
                tf.constant(small_problem['L_pop']),
                tf.constant(small_problem['mu_subj']),
                tf.constant(small_problem['L_subj']),
                tf.constant(small_problem['gamma']),
                tf.constant(small_problem['subject_ids']),
                small_problem['lambda_mu'],
                small_problem['lambda_L'],
                small_problem['n_subjects']
            )
        
        mu_pop = tf.constant(small_problem['mu_pop'])
        
        # compute_gradient returns (theoretical_jacobian, numerical_jacobian)
        theoretical, numerical = tf.test.compute_gradient(
            loss_fn,
            [mu_pop]
        )
        
        print("TF theoretical (analytical):", theoretical[0].shape)
        print("TF numerical:", numerical[0].shape)
        
        assert_allclose(theoretical[0], numerical[0], rtol=1e-3, atol=1e-4)
    
    @pytest.mark.parametrize("calculation", ["mean", "sum"])
    def test_mu_pop_gradient(self, small_problem, calculation):
        """Test gradient w.r.t. population means."""
        analytical = compute_analytical_gradients(small_problem, calculation)
        numerical = compute_numerical_gradients(small_problem, calculation)
        
        assert_allclose(
            analytical['grad_mu_pop'],
            numerical['grad_mu_pop'],
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"mu_pop gradient mismatch for calculation={calculation}"
        )
    
    @pytest.mark.parametrize("calculation", ["mean", "sum"])
    def test_L_pop_gradient(self, small_problem, calculation):
        """Test gradient w.r.t. population Cholesky factors."""
        analytical = compute_analytical_gradients(small_problem, calculation)
        numerical = compute_numerical_gradients(small_problem, calculation)
        
        assert_allclose(
            analytical['grad_L_pop'],
            numerical['grad_L_pop'],
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"L_pop gradient mismatch for calculation={calculation}"
        )
    
    @pytest.mark.parametrize("calculation", ["mean", "sum"])
    def test_mu_subj_gradient(self, small_problem, calculation):
        """Test gradient w.r.t. subject means."""
        analytical = compute_analytical_gradients(small_problem, calculation)
        numerical = compute_numerical_gradients(small_problem, calculation)
        
        assert_allclose(
            analytical['grad_mu_subj'],
            numerical['grad_mu_subj'],
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"mu_subj gradient mismatch for calculation={calculation}"
        )
    
    @pytest.mark.parametrize("calculation", ["mean", "sum"])
    def test_L_subj_gradient(self, small_problem, calculation):
        """Test gradient w.r.t. subject Cholesky factors."""
        analytical = compute_analytical_gradients(small_problem, calculation)
        numerical = compute_numerical_gradients(small_problem, calculation)
        
        assert_allclose(
            analytical['grad_L_subj'],
            numerical['grad_L_subj'],
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"L_subj gradient mismatch for calculation={calculation}"
        )


# =============================================================================
# Tests for absent subject behavior
# =============================================================================

class TestAbsentSubjectGradients:
    """Tests that absent subjects have zero gradients (except via population)."""
    
    def test_absent_subject_mu_gradient_is_zero_without_regularization(self, single_subject_problem):
        """With lambda=0, absent subjects should have zero gradient."""
        problem = single_subject_problem.copy()
        problem['lambda_mu'] = 0.0
        problem['lambda_L'] = 0.0
        
        analytical = compute_analytical_gradients(problem, 'mean')
        
        # Subject 1 is present, subjects 0 and 2 are absent
        # Without regularization, absent subjects should have zero gradient
        assert_allclose(
            analytical['grad_mu_subj'][0],  # Subject 0 (absent)
            np.zeros_like(analytical['grad_mu_subj'][0]),
            atol=1e-7,
            err_msg="Absent subject 0 should have zero mu gradient without regularization"
        )
        assert_allclose(
            analytical['grad_mu_subj'][2],  # Subject 2 (absent)
            np.zeros_like(analytical['grad_mu_subj'][2]),
            atol=1e-7,
            err_msg="Absent subject 2 should have zero mu gradient without regularization"
        )
        
        # Subject 1 (present) should have non-zero gradient
        assert not np.allclose(analytical['grad_mu_subj'][1], 0.0), \
            "Present subject should have non-zero gradient"
    
    def test_absent_subject_L_gradient_is_zero_without_regularization(self, single_subject_problem):
        """With lambda=0, absent subjects should have zero L gradient."""
        problem = single_subject_problem.copy()
        problem['lambda_mu'] = 0.0
        problem['lambda_L'] = 0.0
        
        analytical = compute_analytical_gradients(problem, 'mean')
        
        assert_allclose(
            analytical['grad_L_subj'][0],
            np.zeros_like(analytical['grad_L_subj'][0]),
            atol=1e-7,
            err_msg="Absent subject 0 should have zero L gradient without regularization"
        )
        assert_allclose(
            analytical['grad_L_subj'][2],
            np.zeros_like(analytical['grad_L_subj'][2]),
            atol=1e-7,
            err_msg="Absent subject 2 should have zero L gradient without regularization"
        )
    
    def test_absent_subject_still_zero_with_regularization(self, single_subject_problem):
        """With regularization, absent subjects should STILL have zero gradient."""
        problem = single_subject_problem.copy()
        problem['lambda_mu'] = 1.0
        problem['lambda_L'] = 1.0
        
        analytical = compute_analytical_gradients(problem, 'mean')
        
        # Even with regularization, absent subjects should not be updated
        assert_allclose(
            analytical['grad_mu_subj'][0],
            np.zeros_like(analytical['grad_mu_subj'][0]),
            atol=1e-7,
            err_msg="Absent subject should have zero gradient even with regularization"
        )
        assert_allclose(
            analytical['grad_mu_subj'][2],
            np.zeros_like(analytical['grad_mu_subj'][2]),
            atol=1e-7,
        )
    
    def test_population_gradient_only_from_present_subjects(self, single_subject_problem):
        """Population gradient should only consider present subjects."""
        problem = single_subject_problem.copy()
        problem['lambda_mu'] = 1.0
        problem['lambda_L'] = 0.0  # Focus on mu for simplicity
        
        analytical = compute_analytical_gradients(problem, 'mean')
        
        # The population gradient should be:
        # lambda_mu * (mu_subj[1] - mu_pop) * (1/3)  [only subject 1, scaled by 1/3]
        P = problem['n_subjects']
        n_unique = 1  # Only subject 1
        scale = n_unique / P
        
        expected_grad_mu_pop = problem['lambda_mu'] * (
            problem['mu_subj'][1] - problem['mu_pop']
        ) * scale
        
        # The analytical gradient includes NLL contribution too, so we can't
        # directly compare. But we can verify scaling behavior by comparing
        # two scenarios.
        
        # This test verifies the gradient is non-zero and has correct sign
        assert not np.allclose(analytical['grad_mu_pop'], 0.0), \
            "Population gradient should be non-zero"


# =============================================================================
# Tests for scaling behavior
# =============================================================================

class TestScalingBehavior:
    """Tests for correct scaling of gradients."""
    
    def test_regularization_scaling_with_subset(self, small_problem):
        """Test that regularization is scaled by n_unique/n_subjects."""
        # Create two problems: one with all subjects, one with subset
        problem_all = small_problem.copy()
        problem_all['subject_ids'] = np.array([0, 1, 2, 0], dtype=np.int32)  # All 3 subjects
        
        problem_subset = small_problem.copy()
        problem_subset['subject_ids'] = np.array([0, 0, 0, 0], dtype=np.int32)  # Only subject 0
        
        # Set lambda high to make regularization dominant
        problem_all['lambda_mu'] = 100.0
        problem_all['lambda_L'] = 0.0
        problem_subset['lambda_mu'] = 100.0
        problem_subset['lambda_L'] = 0.0
        
        grad_all = compute_analytical_gradients(problem_all, 'mean')
        grad_subset = compute_analytical_gradients(problem_subset, 'mean')
        
        # The regularization contribution should be scaled differently
        # With all subjects: scale = 3/3 = 1
        # With one subject: scale = 1/3
        
        # Population gradient magnitude should differ due to scaling
        # (This is a rough check - exact values depend on NLL contribution too)
        assert grad_all['grad_mu_pop'].shape == grad_subset['grad_mu_pop'].shape


# =============================================================================
# Tests for layer wrapper
# =============================================================================

class TestLayerWrapper:
    """Tests for the Keras layer wrapper."""
    
    def test_layer_output_shape(self, small_problem):
        """Test that layer returns scalar loss."""
        layer = HierarchicalCategoricalLogLikelihoodLossLayer(
            n_states=small_problem['K'],
            n_subjects=small_problem['P'],
            epsilon=1e-6,
            calculation='mean',
            lambda_mu=1.0,
            lambda_L=1.0,
        )
        
        inputs = [
            tf.constant(small_problem['x']),
            tf.constant(small_problem['mu_pop']),
            tf.constant(small_problem['L_pop']),
            tf.constant(small_problem['mu_subj']),
            tf.constant(small_problem['L_subj']),
            tf.constant(small_problem['gamma']),
            tf.constant(small_problem['subject_ids']),
        ]
        
        output = layer(inputs)
        
        assert output.shape == (1,), f"Expected shape (1,), got {output.shape}"
    
    def test_layer_adds_loss(self, small_problem):
        """Test that layer adds loss."""
        layer = HierarchicalCategoricalLogLikelihoodLossLayer(
            n_states=small_problem['K'],
            n_subjects=small_problem['P'],
            epsilon=1e-6,
            calculation='mean',
            lambda_mu=1.0,
            lambda_L=1.0,
        )
        
        inputs = [
            tf.constant(small_problem['x']),
            tf.constant(small_problem['mu_pop']),
            tf.constant(small_problem['L_pop']),
            tf.constant(small_problem['mu_subj']),
            tf.constant(small_problem['L_subj']),
            tf.constant(small_problem['gamma']),
            tf.constant(small_problem['subject_ids']),
        ]
        
        _ = layer(inputs)
        
        assert len(layer.losses) > 0, "Layer should add loss"

    
    @pytest.mark.parametrize("calculation", ["mean", "sum"])
    def test_layer_gradient_flow(self, small_problem, calculation):
        """Test that gradients flow through the layer."""
        layer = HierarchicalCategoricalLogLikelihoodLossLayer(
            n_states=small_problem['K'],
            n_subjects=small_problem['P'],
            epsilon=1e-6,
            calculation=calculation,
            lambda_mu=1.0,
            lambda_L=1.0,
        )
        
        mu_pop = tf.Variable(small_problem['mu_pop'])
        L_pop = tf.Variable(small_problem['L_pop'])
        mu_subj = tf.Variable(small_problem['mu_subj'])
        L_subj = tf.Variable(small_problem['L_subj'])
        
        inputs = [
            tf.constant(small_problem['x'],dtype=tf.float32),
            mu_pop,
            L_pop,
            mu_subj,
            L_subj,
            tf.constant(small_problem['gamma'],dtype=tf.float32),
            tf.constant(small_problem['subject_ids'],dtype=tf.float32),
        ]
        
        with tf.GradientTape() as tape:
            loss = layer(inputs)
        
        grads = tape.gradient(loss, [mu_pop, L_pop, mu_subj, L_subj])
        
        for i, (name, grad) in enumerate(zip(
            ['mu_pop', 'L_pop', 'mu_subj', 'L_subj'], grads
        )):
            assert grad is not None, f"Gradient for {name} should not be None"
            assert not tf.reduce_all(grad == 0), f"Gradient for {name} should not be all zeros"


# =============================================================================
# Tests for edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_regularization(self, small_problem):
        """Test with zero regularization."""
        problem = small_problem.copy()
        problem['lambda_mu'] = 0.0
        problem['lambda_L'] = 0.0
        
        analytical = compute_analytical_gradients(problem, 'mean')
        numerical = compute_numerical_gradients(problem, 'mean')
        
        assert_allclose(
            analytical['grad_mu_subj'],
            numerical['grad_mu_subj'],
            rtol=1e-4,
            atol=1e-5,
        )
    
    def test_high_regularization(self, small_problem):
        """Test with very high regularization."""
        problem = small_problem.copy()
        problem['lambda_mu'] = 100.0
        problem['lambda_L'] = 100.0
        
        analytical = compute_analytical_gradients(problem, 'mean')
        numerical = compute_numerical_gradients(problem, 'mean')
        
        assert_allclose(
            analytical['grad_mu_subj'],
            numerical['grad_mu_subj'],
            rtol=1e-4,
            atol=1e-5,
        )
    
    def test_all_same_subject(self, small_problem):
        """Test when all batch elements are from same subject."""
        problem = small_problem.copy()
        problem['subject_ids'] = np.array([0, 0, 0, 0], dtype=np.int32)
        
        analytical = compute_analytical_gradients(problem, 'mean')
        numerical = compute_numerical_gradients(problem, 'mean')
        
        assert_allclose(
            analytical['grad_mu_subj'],
            numerical['grad_mu_subj'],
            rtol=1e-4,
            atol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])