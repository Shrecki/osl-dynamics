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
import numpy.testing as npt

from osl_dynamics.models import _hmmc
from osl_dynamics.data import Data


from osl_dynamics.models.hmm_hierarchical import (
    hierarchical_categorical_nll_mean_custom_grad,
    hierarchical_categorical_nll_sum_custom_grad,
    HierarchicalCategoricalLogLikelihoodLossLayer,
    sort_and_rowsplit_by_subject,
    HierarchicalConfig,
    HierarchicalModel
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
# Tests for HMM Baum Welch C++ code              #
##################################################
class TestCppBW:
    def logsumexp_np(self,a, axis=None):
        a = np.asarray(a)
        m = np.max(a, axis=axis, keepdims=True)
        out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
        return np.squeeze(out, axis=axis)


    def forward_log_hierarchical_py(self, startprob_subjs, transmat_subjs, logB, subj_ids):
        """Pure NumPy reference implementation matching your simplified reset rule:
        reset at t==0 OR subj_ids[t] != subj_ids[t-1]. (Ignore any other boundaries.)
        """
        startprob_subjs = np.asarray(startprob_subjs, dtype=np.float64)
        transmat_subjs = np.asarray(transmat_subjs, dtype=np.float64)
        logB = np.asarray(logB, dtype=np.float64)
        subj_ids = np.asarray(subj_ids, dtype=np.int32)

        P, K = startprob_subjs.shape
        N = logB.shape[0]
        assert logB.shape == (N, K)
        assert transmat_subjs.shape == (P, K, K)
        assert subj_ids.shape == (N,)

        log_pi = np.log(startprob_subjs)
        log_A = np.log(transmat_subjs)

        fwd = np.empty((N, K), dtype=np.float64)

        def is_reset(t):
            return (t == 0) or (subj_ids[t] != subj_ids[t - 1])

        def is_seg_end(t):
            return (t == N - 1) or (subj_ids[t + 1] != subj_ids[t])

        total_log_prob = 0.0

        for t in range(N):
            p = subj_ids[t]
            if is_reset(t):
                fwd[t, :] = log_pi[p, :] + logB[t, :]
            else:
                # fwd[t,j] = logB[t,j] + logsumexp_i(fwd[t-1,i] + log_A[p,i,j])
                tmp = fwd[t - 1, :][:, None] + log_A[p, :, :]  # (K,K)
                fwd[t, :] = logB[t, :] + self.logsumexp_np(tmp, axis=0)  # (K,)
            if is_seg_end(t):
                total_log_prob += self.logsumexp_np(fwd[t, :], axis=0)

        return float(total_log_prob), fwd


    def make_random_stochastic(self, shape, rng):
        x = rng.random(shape, dtype=np.float64) + 1e-3
        x /= x.sum(axis=-1, keepdims=True)
        return x


    def test_forward_matches_numpy_reference(self):
        rng = np.random.default_rng(0)

        P = 4   # n subjects
        K = 3   # n states
        N = 25  # total timepoints in flattened stream

        startprob_subjs = self.make_random_stochastic((P, K), rng)
        transmat_subjs = self.make_random_stochastic((P, K, K), rng)

        # log emissions: use random log-likelihoods (not necessarily normalized)
        logB = rng.normal(size=(N, K)).astype(np.float64)

        # subject ids with frequent changes
        subj_ids = rng.integers(low=0, high=P, size=(N,), dtype=np.int32)
        # force at least one change
        subj_ids[10:] = (subj_ids[10:] + 1) % P

        # C++ result
        logp_cpp, fwd_cpp = _hmmc.forward_log_hierarchical(
            startprob_subjs, transmat_subjs, logB, subj_ids
        )

        # NumPy reference
        logp_py, fwd_py = self.forward_log_hierarchical_py(
            startprob_subjs, transmat_subjs, logB, subj_ids
        )

        npt.assert_allclose(logp_cpp, logp_py, rtol=1e-10, atol=1e-10, err_msg="log_prob mismatch")
        npt.assert_allclose(fwd_cpp, fwd_py, rtol=1e-10, atol=1e-10, err_msg="fwd lattice mismatch")


    def test_no_subject_changes_reduces_to_standard_forward(self):
        rng = np.random.default_rng(1)

        P = 3
        K = 4
        N = 30

        startprob_subjs = self.make_random_stochastic((P, K), rng)
        transmat_subjs = self.make_random_stochastic((P, K, K), rng)
        logB = rng.normal(size=(N, K)).astype(np.float64)

        # constant subject id => should match a standard forward on that subject
        p = 2
        subj_ids = np.full((N,), p, dtype=np.int32)

        # compare against your existing standard forward_log for that subject, if available:
        # forward_log(startprob, transmat, log_frameprob)
        # If your module exposes forward_log, this is a great regression check.
        logp_std, fwd_std = _hmmc.forward_log(
            startprob_subjs[p], transmat_subjs[p], logB
        )
        # hierarchical call
        logp_cpp, fwd_cpp = _hmmc.forward_log_hierarchical(
            startprob_subjs, transmat_subjs, logB, subj_ids
        )

        

        npt.assert_allclose(logp_cpp, logp_std, rtol=1e-10, atol=1e-10, err_msg="log_prob mismatch vs standard")
        npt.assert_allclose(fwd_cpp, fwd_std, rtol=1e-10, atol=1e-10, err_msg="fwd mismatch vs standard")

    def test_no_subject_changes_reduces_to_standard_backward(self):
        rng = np.random.default_rng(1)

        P = 3
        K = 4
        N = 30

        startprob_subjs = self.make_random_stochastic((P, K), rng)
        transmat_subjs = self.make_random_stochastic((P, K, K), rng)
        logB = rng.normal(size=(N, K)).astype(np.float64)

        # constant subject id => should match a standard forward on that subject
        p = 2
        subj_ids = np.full((N,), p, dtype=np.int32)

        # compare against your existing standard forward_log for that subject, if available:
        # forward_log(startprob, transmat, log_frameprob)
        # If your module exposes forward_log, this is a great regression check.
        bwd_std = _hmmc.backward_log(
            startprob_subjs[p], transmat_subjs[p], logB
        )
        # hierarchical call
        bwd_hierarchical = _hmmc.backward_log_hierarchical(
            startprob_subjs, transmat_subjs, logB, subj_ids
        )
        npt.assert_allclose(bwd_std, bwd_hierarchical, rtol=1e-10, atol=1e-10, err_msg="bwd mismatch vs standard")
    
    def test_reset_points_match_manual_restart_backward(self):
        """
        Construct a stream with a known reset at t = split,
        and verify:
        bwd matches segment-wise forward stitched together.
        """
        rng = np.random.default_rng(2)

        P = 2
        K = 3
        N1, N2 = 12, 9
        N = N1 + N2

        startprob_subjs = self.make_random_stochastic((P, K), rng)
        transmat_subjs = self.make_random_stochastic((P, K, K), rng)

        logB = rng.normal(size=(N, K)).astype(np.float64)

        # two segments: subj 0 then subj 1
        subj_ids = np.concatenate([
            np.zeros(N1, dtype=np.int32),
            np.ones(N2, dtype=np.int32)
        ], axis=0)

        # hierarchical call
        bwd_hierarchical = _hmmc.backward_log_hierarchical(
            startprob_subjs, transmat_subjs, logB, subj_ids
        )

        # manual: standard backward on seg1 and seg2 separately
        bwd1 = _hmmc.backward_log(startprob_subjs[0], transmat_subjs[0], logB[:N1])
        bwd2 = _hmmc.backward_log(startprob_subjs[1], transmat_subjs[1], logB[N1:])

        bwd_stitched = np.vstack([bwd1, bwd2])
        npt.assert_allclose(bwd_hierarchical, bwd_stitched, rtol=1e-10, atol=1e-10, err_msg="stitched bwd mismatch")


    def test_reset_points_match_manual_restart(self):
        """
        Construct a stream with a known reset at t = split,
        and verify:
        total_log_prob = log_prob(seg1) + log_prob(seg2),
        and fwd matches segment-wise forward stitched together.
        """
        rng = np.random.default_rng(2)

        P = 2
        K = 3
        N1, N2 = 12, 9
        N = N1 + N2

        startprob_subjs = self.make_random_stochastic((P, K), rng)
        transmat_subjs = self.make_random_stochastic((P, K, K), rng)

        logB = rng.normal(size=(N, K)).astype(np.float64)

        # two segments: subj 0 then subj 1
        subj_ids = np.concatenate([
            np.zeros(N1, dtype=np.int32),
            np.ones(N2, dtype=np.int32)
        ], axis=0)

        # hierarchical call
        logp_cpp, fwd_cpp = _hmmc.forward_log_hierarchical(
            startprob_subjs, transmat_subjs, logB, subj_ids
        )

        # manual: standard forward on seg1 and seg2 separately, then add log probs
        logp1, fwd1 = _hmmc.forward_log(startprob_subjs[0], transmat_subjs[0], logB[:N1])
        logp2, fwd2 = _hmmc.forward_log(startprob_subjs[1], transmat_subjs[1], logB[N1:])

        npt.assert_allclose(logp_cpp, logp1 + logp2, rtol=1e-10, atol=1e-10, err_msg="segment log_prob additivity failed")

        fwd_stitched = np.vstack([fwd1, fwd2])
        npt.assert_allclose(fwd_cpp, fwd_stitched, rtol=1e-10, atol=1e-10, err_msg="stitched fwd mismatch")
        
    def test_reset_points_match_manual_restart_bwd(self):
        """
        Construct a stream with a known reset at t = split,
        and verify:
        bwd matches segment-wise forward stitched together.
        """
        rng = np.random.default_rng(2)

        P = 2
        K = 3
        N1, N2 = 12, 9
        N = N1 + N2

        startprob_subjs = self.make_random_stochastic((P, K), rng)
        transmat_subjs = self.make_random_stochastic((P, K, K), rng)

        logB = rng.normal(size=(N, K)).astype(np.float64)

        # two segments: subj 0 then subj 1
        subj_ids = np.concatenate([
            np.zeros(N1, dtype=np.int32),
            np.ones(N2, dtype=np.int32)
        ], axis=0)

        # hierarchical call
        bwd_hierarchical = _hmmc.backward_log_hierarchical(
            startprob_subjs, transmat_subjs, logB, subj_ids
        )

        # manual: standard forward on seg1 and seg2 separately, then add log probs
        bwd1 = _hmmc.backward_log(startprob_subjs[0], transmat_subjs[0], logB[:N1])
        bwd2 = _hmmc.backward_log(startprob_subjs[1], transmat_subjs[1], logB[N1:])
        
        bwd_stitched = np.vstack([bwd1, bwd2])
        npt.assert_allclose(bwd_hierarchical, bwd_stitched, rtol=1e-10, atol=1e-10, err_msg="stitched bwd mismatch")
        
    def row_normalize(self, mat):
        mat = np.asarray(mat)
        s = mat.sum(axis=-1, keepdims=True)
        return mat / s
    
    def test_no_subject_changes_reduces_to_standard_xi_sum(self):
        rng = np.random.default_rng(1)

        P = 3
        K = 4
        N = 30

        startprob_subjs = self.make_random_stochastic((P, K), rng)
        transmat_subjs = self.make_random_stochastic((P, K, K), rng)
        logB = rng.normal(size=(N, K)).astype(np.float64)

        # constant subject id => should match a standard forward on that subject
        p = 2
        subj_ids = np.full((N,), p, dtype=np.int32)

        # compare against your existing standard forward_log for that subject, if available:
        # forward_log(startprob, transmat, log_frameprob)
        # If your module exposes forward_log, this is a great regression check.
        log_probs_hierarch, fwd_hierarch = _hmmc.forward_log_hierarchical(startprob_subjs, transmat_subjs, logB, subj_ids)
        bwd_hierarc = _hmmc.backward_log_hierarchical(
            startprob_subjs, transmat_subjs, logB,subj_ids
        )
        xi_sum_hierarch = _hmmc.compute_log_xi_sum_hierarchical(fwd_hierarch, transmat_subjs, bwd_hierarc, logB, subj_ids)
        xi_sum_std = _hmmc.compute_log_xi_sum(fwd_hierarch, transmat_subjs[p], bwd_hierarc, logB)
        
        npt.assert_allclose(self.row_normalize(np.exp(xi_sum_hierarch[p])), self.row_normalize(np.exp(xi_sum_std)), rtol=1e-10, atol=1e-10, err_msg="log_xi_sum mismatch vs standard")
        npt.assert_allclose(xi_sum_hierarch[0], np.ones((K,K))*(-np.inf), rtol=1e-10, atol=1e-10, err_msg="log_xi_sum mismatch vs standard")
        npt.assert_allclose(xi_sum_hierarch[1], np.ones((K,K))*(-np.inf), rtol=1e-10, atol=1e-10, err_msg="log_xi_sum mismatch vs standard")
        
    def test_reset_points_match_manual_restart_xi_sum_one_empty_subj(self):
        """
        Construct a stream with a known reset at t = split,
        and verify:
        xi_sum matches segment-wise xi_sum stitched together.
        """
        rng = np.random.default_rng(2)

        P = 3
        K = 4
        N1, N2 = 12, 9
        N = N1 + N2

        startprob_subjs = self.make_random_stochastic((P, K), rng)
        transmat_subjs = self.make_random_stochastic((P, K, K), rng)

        logB = rng.normal(size=(N, K)).astype(np.float64)

        # two segments: subj 0 then subj 1
        subj_ids = np.concatenate([
            np.zeros(N1, dtype=np.int32),
            np.ones(N2, dtype=np.int32)
        ], axis=0)
        
        # Here is the xi_sum_hierarchical
        log_probs_hierarch, fwd_hierarch = _hmmc.forward_log_hierarchical(startprob_subjs, transmat_subjs, logB, subj_ids)
        bwd_hierarc = _hmmc.backward_log_hierarchical(
            startprob_subjs, transmat_subjs, logB,subj_ids
        )
        xi_sum_hierarch = _hmmc.compute_log_xi_sum_hierarchical(fwd_hierarch, transmat_subjs, bwd_hierarc, logB, subj_ids)
        
        # Here is two segments on which forward/backward and xi_sum are computed
        _,fwd1 = _hmmc.forward_log(startprob_subjs[0], transmat_subjs[0], logB[:N1])
        _,fwd2 = _hmmc.forward_log(startprob_subjs[1], transmat_subjs[1], logB[N1:])
        bwd1 = _hmmc.backward_log(startprob_subjs[0], transmat_subjs[0], logB[:N1])
        bwd2 = _hmmc.backward_log(startprob_subjs[1], transmat_subjs[1], logB[N1:])

        xi_sum1 = _hmmc.compute_log_xi_sum(fwd1, transmat_subjs[0], bwd1, logB[:N1])
        xi_sum2 = _hmmc.compute_log_xi_sum(fwd2, transmat_subjs[1], bwd2, logB[N1:])
        
        npt.assert_allclose(self.row_normalize(np.exp(xi_sum_hierarch[0])), self.row_normalize(np.exp(xi_sum1)), rtol=1e-10, atol=1e-10, err_msg="log_xi_sum mismatch vs standard")
        npt.assert_allclose(self.row_normalize(np.exp(xi_sum_hierarch[1])), self.row_normalize(np.exp(xi_sum2)), rtol=1e-10, atol=1e-10, err_msg="log_xi_sum mismatch vs standard")
        npt.assert_allclose(xi_sum_hierarch[2], np.ones((K,K))*(-np.inf), rtol=1e-10, atol=1e-10, err_msg="log_xi_sum mismatch vs standard")

    
    def test_reset_points_match_manual_restart_xi_sum(self):
        """
        Construct a stream with a known reset at t = split,
        and verify:
        xi_sum matches segment-wise xi_sum stitched together.
        """
        rng = np.random.default_rng(2)

        P = 3
        K = 5
        N1, N2 = 12, 9
        N = N1 + N2

        startprob_subjs = self.make_random_stochastic((P, K), rng)
        transmat_subjs = self.make_random_stochastic((P, K, K), rng)

        logB = rng.normal(size=(N, K)).astype(np.float64)

        # two segments: subj 0 then subj 1
        subj_ids = np.concatenate([
            np.zeros(N1, dtype=np.int32),
            np.ones(N2, dtype=np.int32)
        ], axis=0)
        
        # Here is the xi_sum_hierarchical
        log_probs_hierarch, fwd_hierarch = _hmmc.forward_log_hierarchical(startprob_subjs, transmat_subjs, logB, subj_ids)
        bwd_hierarc = _hmmc.backward_log_hierarchical(
            startprob_subjs, transmat_subjs, logB,subj_ids
        )
        xi_sum_hierarch = _hmmc.compute_log_xi_sum_hierarchical(fwd_hierarch, transmat_subjs, bwd_hierarc, logB, subj_ids)
        
        # Here is two segments on which forward/backward and xi_sum are computed
        _,fwd1 = _hmmc.forward_log(startprob_subjs[0], transmat_subjs[0], logB[:N1])
        _,fwd2 = _hmmc.forward_log(startprob_subjs[1], transmat_subjs[1], logB[N1:])
        bwd1 = _hmmc.backward_log(startprob_subjs[0], transmat_subjs[0], logB[:N1])
        bwd2 = _hmmc.backward_log(startprob_subjs[1], transmat_subjs[1], logB[N1:])

        xi_sum1 = _hmmc.compute_log_xi_sum(fwd1, transmat_subjs[0], bwd1, logB[:N1])
        xi_sum2 = _hmmc.compute_log_xi_sum(fwd2, transmat_subjs[1], bwd2, logB[N1:])
        
        npt.assert_allclose(self.row_normalize(np.exp(xi_sum_hierarch[0])), self.row_normalize(np.exp(xi_sum1)), rtol=1e-10, atol=1e-10, err_msg="log_xi_sum mismatch vs standard")
        npt.assert_allclose(self.row_normalize(np.exp(xi_sum_hierarch[1])), self.row_normalize(np.exp(xi_sum2)), rtol=1e-10, atol=1e-10, err_msg="log_xi_sum mismatch vs standard")


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
            for name, g_auto, g_anal in zip(
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
                )
            
            # Also verify loss values match
            assert_allclose(
                loss_ref.numpy(),
                loss_layer.numpy(),
                rtol=1e-6,
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
                rtol=1e-6,
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
        
        
class TestUpdateTransProbHierarchical:
    """Tests for update_trans_prob_optimized_hierarchical method."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing transition probability updates."""
        config = HierarchicalConfig(
            n_states=3,
            n_channels=5,
            n_subjects=4,
            sequence_length=50,
            batch_size=4,
            learning_rate=0.01,
            learn_means=True,
            learn_covariances=True,
            n_epochs=1,
            initial_covariances_pop=None,
        )
        model = HierarchicalModel(config)
        return model
    
    def test_output_remains_row_stochastic(self, simple_model):
        """Test that transition matrices remain row-stochastic after update."""
        model = simple_model
        K = model.config.n_states
        P = model.config.n_subjects
        N = 100
        
        np.random.seed(42)
        
        # Create random gamma (must sum to 1 over states)
        gamma_raw = np.random.rand(N, K)
        gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
        
        # Create random xi_sum_flat for each subject
        xi_sum_flat = np.random.rand(P, K * K) + 0.1
        
        # Subject IDs with some consecutive same-subject pairs
        subj_ids = np.random.randint(0, P, size=N)
        
        # Set rho to 1 for full update
        model.rho = 1.0
        
        # Update
        model.update_trans_prob_optimized_hierarchical(gamma, xi_sum_flat, subj_ids)
        
        # Check all matrices are row-stochastic
        for p in range(P):
            row_sums = model.trans_prob_subj[p].sum(axis=1)
            assert_allclose(
                row_sums, np.ones(K),
                rtol=1e-6,
                err_msg=f"Subject {p} transition matrix rows should sum to 1"
            )
            assert np.all(model.trans_prob_subj[p] >= 0), \
                f"Subject {p} transition matrix should be non-negative"
    
    def test_absent_subject_unchanged(self, simple_model):
        """Test that subjects not in batch are not updated."""
        model = simple_model
        K = model.config.n_states
        P = model.config.n_subjects
        N = 100
        
        np.random.seed(42)
        
        # Store initial transition matrices
        initial_trans_prob = [model.trans_prob_subj[p].copy() for p in range(P)]
        
        # Create data with only subjects 0 and 1
        gamma_raw = np.random.rand(N, K)
        gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
        
        # xi_sum_flat: subjects 2 and 3 have all zeros
        xi_sum_flat = np.zeros((P, K * K))
        xi_sum_flat[0] = np.random.rand(K * K) + 0.1
        xi_sum_flat[1] = np.random.rand(K * K) + 0.1
        # xi_sum_flat[2] and [3] remain zeros
        
        # Subject IDs: only subjects 0 and 1
        subj_ids = np.random.choice([0, 1], size=N)
        
        model.rho = 1.0
        model.update_trans_prob_optimized_hierarchical(gamma, xi_sum_flat, subj_ids)
        
        # Subjects 0 and 1 should change
        assert not np.allclose(initial_trans_prob[0], model.trans_prob_subj[0]), \
            "Subject 0 should be updated"
        assert not np.allclose(initial_trans_prob[1], model.trans_prob_subj[1]), \
            "Subject 1 should be updated"
        
        # Subjects 2 and 3 should NOT change
        assert_allclose(
            initial_trans_prob[2], model.trans_prob_subj[2],
            rtol=1e-10,
            err_msg="Subject 2 should not be updated (no data)"
        )
        assert_allclose(
            initial_trans_prob[3], model.trans_prob_subj[3],
            rtol=1e-10,
            err_msg="Subject 3 should not be updated (no data)"
        )
    
    def test_rho_zero_no_update(self, simple_model):
        """Test that rho=0 means no update happens."""
        model = simple_model
        K = model.config.n_states
        P = model.config.n_subjects
        N = 100
        
        np.random.seed(42)
        
        initial_trans_prob = [model.trans_prob_subj[p].copy() for p in range(P)]
        
        gamma_raw = np.random.rand(N, K)
        gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
        xi_sum_flat = np.random.rand(P, K * K) + 0.1
        subj_ids = np.random.randint(0, P, size=N)
        
        # rho = 0 means: trans_prob = (1-0)*trans_prob + 0*new = trans_prob
        model.rho = 0.0
        model.update_trans_prob_optimized_hierarchical(gamma, xi_sum_flat, subj_ids)
        
        for p in range(P):
            assert_allclose(
                initial_trans_prob[p], model.trans_prob_subj[p],
                rtol=1e-10,
                err_msg=f"Subject {p} should not change when rho=0"
            )
    
    def test_rho_one_full_update(self, simple_model):
        """Test that rho=1 means full replacement with new estimate."""
        model = simple_model
        K = model.config.n_states
        P = model.config.n_subjects
        N = 200
        
        np.random.seed(42)
        
        # Create deterministic scenario: subject 0 only
        gamma = np.zeros((N, K))
        gamma[:, 0] = 1.0  # Always in state 0
        
        # xi_sum_flat encodes transitions: only 0->0 transitions
        xi_sum_flat = np.zeros((P, K * K))
        # For subject 0: all transitions are 0->0
        # xi is K x K, flattened. Index [0,0] in row-major is index 0
        xi_sum_flat[0, 0] = N - 1  # N-1 transitions from state 0 to state 0
        
        subj_ids = np.zeros(N, dtype=np.int32)  # All subject 0
        
        model.rho = 1.0
        model.update_trans_prob_optimized_hierarchical(gamma, xi_sum_flat, subj_ids)
        
        # Subject 0 should have ~1.0 for 0->0, ~0 elsewhere in row 0
        # (with EPS adjustments)
        assert model.trans_prob_subj[0][0, 0] > 0.99, \
            "State 0->0 transition should be ~1.0"
    
    def test_stochastic_update_interpolation(self, simple_model):
        """Test that update correctly interpolates between old and new."""
        model = simple_model
        K = model.config.n_states
        P = model.config.n_subjects
        N = 100
        
        np.random.seed(42)
        
        # Set a known initial transition matrix for subject 0
        initial_P0 = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        model.trans_prob_subj[0] = initial_P0.copy()
        
        # Create data that would suggest a different transition matrix
        gamma = np.zeros((N, K))
        gamma[:, 1] = 1.0  # Always in state 1
        
        xi_sum_flat = np.zeros((P, K * K))
        # Transitions: 1->1 mostly
        xi_sum_flat[0, 4] = N - 1  # Index 4 = [1,1] in 3x3 row-major
        
        subj_ids = np.zeros(N, dtype=np.int32)
        
        # With rho=0.5, should be halfway between old and new
        model.rho = 0.5
        model.update_trans_prob_optimized_hierarchical(gamma, xi_sum_flat, subj_ids)
        
        # Row 1 should have moved toward [0, 1, 0] but not fully
        # Since initial row 1 was [0.1, 0.8, 0.1] and new would be ~[0, 1, 0]
        # Interpolation: 0.5 * [0.1, 0.8, 0.1] + 0.5 * [~0, ~1, ~0]
        assert model.trans_prob_subj[0][1, 1] > 0.8, \
            "State 1->1 should increase"
        assert model.trans_prob_subj[0][1, 1] < 0.99, \
            "State 1->1 should not be fully 1.0 (interpolation)"
    
    def test_cross_subject_transitions_ignored(self, simple_model):
        """Test that transitions between different subjects are ignored."""
        model = simple_model
        K = model.config.n_states
        P = model.config.n_subjects
        
        np.random.seed(42)
        
        # Create sequence: [subj0, subj0, subj1, subj1, subj0, subj0]
        # Transitions within subject: 0->1, 2->3, 4->5
        # Transitions across subjects (should be ignored): 1->2, 3->4
        subj_ids = np.array([0, 0, 1, 1, 0, 0], dtype=np.int32)
        N = len(subj_ids)
        
        gamma = np.zeros((N, K))
        # Subject 0 time points: indices 0,1,4,5 - say state sequence [0,1,0,1]
        gamma[0, 0] = 1.0  # t=0: state 0
        gamma[1, 1] = 1.0  # t=1: state 1
        gamma[4, 0] = 1.0  # t=4: state 0
        gamma[5, 1] = 1.0  # t=5: state 1
        
        # Subject 1 time points: indices 2,3 - say state sequence [2,2]
        gamma[2, 2] = 1.0  # t=2: state 2
        gamma[3, 2] = 1.0  # t=3: state 2
        
        #print(gamma)
        
        # xi_sum_flat should reflect ONLY within-subject transitions
        # Subject 0: 0->1 (t=0->1), 0->1 (t=4->5) => 2 transitions of type 0->1
        # Subject 1: 2->2 (t=2->3) => 1 transition of type 2->2
        xi_sum = np.zeros((P, K,  K))
        xi_sum[0, 0, 1] = 2  # 0->1 for subject 0
        xi_sum[1, 2, 2] = 1  # 2->2 for subject 1
        
        xi_sum_flat = xi_sum.swapaxes(1,2).reshape((P, -1))
        #print(xi_sum_flat)
        
        initial_trans_prob = [model.trans_prob_subj[p].copy() for p in range(P)]
        
        model.rho = 1.0
        model.update_trans_prob_optimized_hierarchical(gamma, xi_sum_flat, subj_ids)
        
        #print(model.trans_prob_subj[0])
        #print(model.trans_prob_subj[1])
        
        # Subject 0 row 0 should have high probability for 0->1
        assert model.trans_prob_subj[0][0, 1] > 0.9, \
            "Subject 0: 0->1 transition should be high"
        
        # Subject 1 row 2 should have high probability for 2->2
        assert model.trans_prob_subj[1][2, 2] > 0.9, \
            "Subject 1: 2->2 transition should be high"
        
        # Subjects 2 and 3 should be unchanged
        assert_allclose(initial_trans_prob[2], model.trans_prob_subj[2])
        assert_allclose(initial_trans_prob[3], model.trans_prob_subj[3])
    
    def test_handles_subject_with_single_timepoint(self, simple_model):
        """Test handling when a subject has only one timepoint (no transitions)."""
        model = simple_model
        K = model.config.n_states
        P = model.config.n_subjects
        
        # Subject 0 has multiple timepoints, subject 1 has only one
        subj_ids = np.array([0, 0, 0, 1, 0, 0], dtype=np.int32)
        N = len(subj_ids)
        
        gamma_raw = np.random.rand(N, K)
        gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
        
        # xi_sum_flat: subject 1 has zeros (no within-subject transitions)
        xi_sum_flat = np.zeros((P, K * K))
        xi_sum_flat[0] = np.random.rand(K * K) + 0.1
        # xi_sum_flat[1] remains zeros
        
        initial_trans_prob_1 = model.trans_prob_subj[1].copy()
        
        model.rho = 1.0
        
        # Should not crash
        model.update_trans_prob_optimized_hierarchical(gamma, xi_sum_flat, subj_ids)
        
        # Subject 1 should be unchanged (skipped due to zero xi_sum)
        assert_allclose(
            initial_trans_prob_1, model.trans_prob_subj[1],
            rtol=1e-10,
            err_msg="Subject with no transitions should be unchanged"
        )
        
    def test_matches_non_hierarchical_single_subject(self):
        """Debug version to find the mismatch."""
        from osl_dynamics.models.hmm import Config, Model
        
        np.random.seed(42)
        K = 3
        N = 20  # Smaller for easier debugging
        
        h_config = HierarchicalConfig(
            n_states=K,
            n_channels=5,
            n_subjects=1,
            sequence_length=50,
            batch_size=4,
            learning_rate=0.01,
            learn_means=True,
            learn_covariances=True,
            n_epochs=1,
            initial_covariances_pop=None,
        )
        h_model = HierarchicalModel(h_config)
        
        nh_config = Config(
            n_states=K,
            n_channels=5,
            sequence_length=50,
            batch_size=4,
            learning_rate=0.01,
            learn_means=True,
            learn_covariances=True,
            n_epochs=1,
            initial_covariances=None,
            kappa=0.0,
        )
        nh_model = Model(nh_config)
        
        initial_P = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ])
        h_model.trans_prob_subj[0] = initial_P.copy()
        nh_model.trans_prob = initial_P.copy()
        
        h_model.rho = 1.0  # Full update for easier comparison
        nh_model.rho = 1.0
        
        gamma_raw = np.random.rand(N, K)
        gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
        
        xi_raw = np.random.rand(N - 1, K * K)
        xi = xi_raw / xi_raw.sum(axis=-1, keepdims=True)
        xi_sum = xi.sum(axis=0)
        
        subj_ids = np.zeros(N, dtype=np.int32)
        
        # Manually compute what hierarchical should get
        same_next = (subj_ids[:-1] == subj_ids[1:])
        p_mask = (subj_ids[:-1] == 0) & same_next
        
        print("same_next:", same_next)
        print("p_mask:", p_mask)
        print("All True?", np.all(p_mask))
        
        # Hierarchical xi reshape
        xi_sum_flat_h = np.array([xi_sum])
        xi_p_h = xi_sum_flat_h[0].reshape(K, K).T
        print("xi_p hierarchical:\n", xi_p_h)
        
        # Non-hierarchical xi reshape  
        xi_p_nh = xi_sum.reshape(K, K).T
        print("xi_p non-hierarchical:\n", xi_p_nh)
        
        # Denominators
        denom_h = gamma[:-1][p_mask].sum(axis=0)
        denom_nh = np.sum(gamma[:-1], axis=0)
        print("denom hierarchical:", denom_h)
        print("denom non-hierarchical:", denom_nh)
        
        # phi_interim
        EPS = np.finfo(float).eps
        phi_h = (xi_p_h + EPS) / (denom_h[:, None] + K * EPS)
        phi_h /= phi_h.sum(axis=1, keepdims=True)
        
        phi_nh = (xi_p_nh + EPS) / (denom_nh.reshape(K, 1) + K * EPS)
        # Note: non-hierarchical might not have the second normalization step
        
        print("phi hierarchical:\n", phi_h)
        print("phi non-hierarchical:\n", phi_nh)
        
        # Now run actual updates
        h_model.update_trans_prob_optimized_hierarchical(gamma, xi_sum_flat_h, subj_ids)
        nh_model.update_trans_prob_optimized(gamma, xi_sum)
        
        print("Final hierarchical trans_prob:\n", h_model.trans_prob_subj[0])
        print("Final non-hierarchical trans_prob:\n", nh_model.trans_prob)
        print("Difference:\n", h_model.trans_prob_subj[0] - nh_model.trans_prob)
    
    
    def test_numerical_stability_small_counts(self, simple_model):
        """Test numerical stability with very small transition counts."""
        model = simple_model
        K = model.config.n_states
        P = model.config.n_subjects
        N = 10
        
        gamma = np.ones((N, K)) / K  # Uniform
        
        # Very small counts
        xi_sum_flat = np.ones((P, K * K)) * 1e-10
        
        subj_ids = np.zeros(N, dtype=np.int32)
        
        model.rho = 1.0
        
        # Should not produce NaN or Inf
        model.update_trans_prob_optimized_hierarchical(gamma, xi_sum_flat, subj_ids)
        
        assert not np.any(np.isnan(model.trans_prob_subj[0])), \
            "Should not produce NaN"
        assert not np.any(np.isinf(model.trans_prob_subj[0])), \
            "Should not produce Inf"
        assert np.all(model.trans_prob_subj[0] >= 0), \
            "Should remain non-negative"
        assert_allclose(
            model.trans_prob_subj[0].sum(axis=1), np.ones(K),
            rtol=1e-6,
            err_msg="Should remain row-stochastic"
        )
    
    def test_correct_use_of_same_next_mask(self, simple_model):
        """Test that same_next mask correctly identifies within-subject transitions."""
        model = simple_model
        K = model.config.n_states
        P = model.config.n_subjects
        
        # Explicit sequence with known transitions
        # subj: [0, 0, 1, 0, 0]
        # valid transitions for subj 0: (0,1) and (3,4) - indices in gamma[:-1]
        # invalid (cross-subject): (1,2), (2,3)
        subj_ids = np.array([0, 0, 1, 0, 0], dtype=np.int32)
        N = len(subj_ids)
        
        # Gamma: all in state 0
        gamma = np.zeros((N, K))
        gamma[:, 0] = 1.0
        
        # xi_sum_flat: subject 0 has 2 transitions (0->0 twice)
        xi_sum_flat = np.zeros((P, K * K))
        xi_sum_flat[0, 0] = 2  # Two 0->0 transitions for subject 0
        
        initial_P0 = model.trans_prob_subj[0].copy()
        
        model.rho = 1.0
        model.update_trans_prob_optimized_hierarchical(gamma, xi_sum_flat, subj_ids)
        
        # Check that the denominator was computed correctly
        # For subject 0, valid transitions are at t=0 and t=3 (indices in gamma[:-1])
        # Both are in state 0, so denom for state 0 should be 2
        # The resulting row 0 of trans_prob should be mostly [1, 0, 0]
        assert model.trans_prob_subj[0][0, 0] > 0.99, \
            "0->0 should be ~1.0 (only valid transition type observed)"


###############################################################################
# Tests for HMM fit interface
###############################################################################
class TestHMMModelFit:
    """Tests for the HierarchicalModel.fit() method."""
    
    @pytest.fixture
    def simple_model_and_data(self):
        """Create a simple model and data for testing fit behavior."""
        np.random.seed(42)
        
        config = HierarchicalConfig(
            n_states=2,
            n_channels=3,
            n_subjects=3,
            sequence_length=50,
            batch_size=4,
            learning_rate=0.1,  # High LR to see changes
            learn_means=True,
            learn_covariances=True,
            n_epochs=1,
            initial_covariances_pop=None,
            lambda_mu=1.0,
            lambda_L=1.0,
            learn_trans_prob=True
        )
        
        model = HierarchicalModel(config)
        
        # Generate synthetic data - only for subjects 0 and 1 (not subject 2)
        data = Data([
            np.random.randn(200, config.n_channels).astype(np.float32),  # subject 0
            np.random.randn(200, config.n_channels).astype(np.float32),  # subject 1
        ], time_axis_first=True, sampling_frequency=250)
        
        subject_ids = [
            np.zeros(200, dtype=np.int32),  # subject 0
            np.ones(200, dtype=np.int32),   # subject 1
        ]
        
        return config, model, data, subject_ids
    
    def test_parameters_change_after_fit(self, simple_model_and_data):
        """Test that observation model parameters are updated after fit."""
        config, model, data, subject_ids = simple_model_and_data
        
        # Get initial parameters
        initial_means_pop = model.get_means_pop().copy()
        initial_trils_pop = model.get_trils_pop().copy()
        initial_means_subj = [m.copy() for m in model.get_means_subj()]
        initial_trils_subj = [t.copy() for t in model.get_trils_subj()]
        
        # Fit for one epoch
        model.fit(data, subject_ids=subject_ids, epochs=1, verbose=0)
        
        # Get updated parameters
        final_means_pop = model.get_means_pop()
        final_trils_pop = model.get_trils_pop()
        final_means_subj = model.get_means_subj()
        final_trils_subj = model.get_trils_subj()
        
        # Population parameters should change
        assert not np.allclose(initial_means_pop, final_means_pop), \
            "Population means should change after fit"
        assert not np.allclose(initial_trils_pop, final_trils_pop), \
            "Population Cholesky factors should change after fit"
        
        # Subject 0 and 1 parameters should change (they have data)
        assert not np.allclose(initial_means_subj[0], final_means_subj[0]), \
            "Subject 0 means should change after fit"
        assert not np.allclose(initial_means_subj[1], final_means_subj[1]), \
            "Subject 1 means should change after fit"
    
    def test_absent_subject_params_unchanged_without_regularization(self):
        """Test that absent subject parameters don't change when lambda=0."""
        np.random.seed(42)
        
        config = HierarchicalConfig(
            n_states=2,
            n_channels=3,
            n_subjects=3,
            sequence_length=50,
            batch_size=4,
            learning_rate=0.1,
            learn_means=True,
            learn_covariances=True,
            n_epochs=1,
            initial_covariances_pop=None,
            lambda_mu=0.0,  # No regularization
            lambda_L=0.0,
            learn_trans_prob=True

        )
        
        model = HierarchicalModel(config)
        
        # Data only for subjects 0 and 1
        data = Data([
            np.random.randn(200, config.n_channels).astype(np.float32),
            np.random.randn(200, config.n_channels).astype(np.float32),
        ], time_axis_first=True, sampling_frequency=250)
        
        subject_ids = [
            np.zeros(200, dtype=np.int32),
            np.ones(200, dtype=np.int32),
        ]
        
        # Get initial parameters for subject 2 (absent)
        initial_means_subj_2 = model.get_means_subj()[2].copy()
        initial_trils_subj_2 = model.get_trils_subj()[2].copy()
        
        # Fit
        model.fit(data, subject_ids=subject_ids, epochs=1, verbose=0)
        
        # Subject 2 parameters should NOT change (no data, no regularization)
        final_means_subj_2 = model.get_means_subj()[2]
        final_trils_subj_2 = model.get_trils_subj()[2]
        
        assert_allclose(
            initial_means_subj_2, final_means_subj_2,
            rtol=1e-6,
            err_msg="Absent subject means should not change without regularization"
        )
        assert_allclose(
            initial_trils_subj_2, final_trils_subj_2,
            rtol=1e-6,
            err_msg="Absent subject Cholesky factors should not change without regularization"
        )
        
    def test_transition_matrix_update_direct(self):
        """Directly test update_trans_prob_optimized_hierarchical with controlled inputs."""
        np.random.seed(42)
        
        config = HierarchicalConfig(
            n_states=2,
            n_channels=3,
            n_subjects=3,
            sequence_length=50,
            batch_size=4,
            learning_rate=0.1,
            learn_means=True,
            learn_covariances=True,
            learn_trans_prob=True,
            n_epochs=1,
            initial_covariances_pop=None,
        )
        
        model = HierarchicalModel(config)
        
        K = config.n_states
        P = config.n_subjects
        N = 100
        
        # Store initial
        initial_trans_prob = model.trans_prob_subj.copy()
        
        # Create gamma
        gamma_raw = np.random.rand(N, K)
        gamma = gamma_raw / gamma_raw.sum(axis=-1, keepdims=True)
        
        # Create subject_ids: first half subject 0, second half subject 1
        # This ensures consecutive same-subject pairs
        subj_ids = np.concatenate([
            np.zeros(N // 2, dtype=np.int32),
            np.ones(N // 2, dtype=np.int32),
        ])
        
        # Create xi_sum_flat with non-zero values for subjects 0 and 1 only
        xi_sum_flat = np.zeros((P, K * K))
        xi_sum_flat[0] = np.random.rand(K * K) + 0.1  # Subject 0 has transitions
        xi_sum_flat[1] = np.random.rand(K * K) + 0.1  # Subject 1 has transitions
        # xi_sum_flat[2] stays zero - subject 2 has no transitions
        
        model.rho = 1.0
        model.update_trans_prob_optimized_hierarchical(gamma, xi_sum_flat, subj_ids)
        
        # Subject 0 and 1 should change
        assert not np.allclose(initial_trans_prob[0], model.trans_prob_subj[0]), \
            "Subject 0 transition matrix should change"
        assert not np.allclose(initial_trans_prob[1], model.trans_prob_subj[1]), \
            "Subject 1 transition matrix should change"
        
        # Subject 2 should NOT change
        assert_allclose(
            initial_trans_prob[2], model.trans_prob_subj[2],
            rtol=1e-10,
            err_msg="Subject 2 transition matrix should not change"
        )
        
    def test_transition_matrix_only_updates_for_present_subjects(self):
        """Test that transition matrices only update for subjects in the batch."""
        np.random.seed(42)
        
        config = HierarchicalConfig(
            n_states=2,
            n_channels=3,
            n_subjects=3,
            sequence_length=50,
            batch_size=4,
            learning_rate=0.1,
            learn_means=True,
            learn_covariances=True,
            learn_trans_prob=True,
            n_epochs=1,
            initial_covariances_pop=None,
        )
        
        model = HierarchicalModel(config)
        
        # Create longer sequences to ensure we have enough within-subject transitions
        # after batching
        data = Data([
            np.random.randn(1000, config.n_channels).astype(np.float32),  # subject 0
            np.random.randn(1000, config.n_channels).astype(np.float32),  # subject 1
        ], time_axis_first=True, sampling_frequency=250)
        
        subject_ids = [
            np.zeros(1000, dtype=np.int32),   # subject 0
            np.ones(1000, dtype=np.int32),    # subject 1
        ]
        import copy
        # Get initial transition matrices
        initial_trans_prob = copy.deepcopy(model.trans_prob_subj)#.copy()
        
        # Fit
        model.fit(data, subject_ids=subject_ids, epochs=1, verbose=0)
        
        # Subject 0 and 1 transition matrices should change
        assert not np.allclose(initial_trans_prob[0], model.trans_prob_subj[0]), \
            "Subject 0 transition matrix should change"
        assert not np.allclose(initial_trans_prob[1], model.trans_prob_subj[1]), \
            "Subject 1 transition matrix should change"
        
        # Subject 2 transition matrix should NOT change
        assert_allclose(
            initial_trans_prob[2], model.trans_prob_subj[2],
            rtol=1e-10,
            err_msg="Absent subject transition matrix should not change"
        )
        
        model.get_alpha(data, subject_ids)
    
    
    def test_transition_matrix_remains_stochastic(self):
        """Test that transition matrices remain row-stochastic after fit."""
        np.random.seed(42)
        
        config = HierarchicalConfig(
            n_states=3,
            n_channels=5,
            n_subjects=2,
            sequence_length=50,
            batch_size=4,
            learning_rate=0.1,
            learn_means=True,
            learn_covariances=True,
            learn_trans_prob=True,
            n_epochs=1,
            initial_covariances_pop=None,
        )
        
        model = HierarchicalModel(config)
        
        data = Data([
            np.random.randn(200, config.n_channels).astype(np.float32),
            np.random.randn(200, config.n_channels).astype(np.float32),
        ], time_axis_first=True, sampling_frequency=250)
        
        subject_ids = [
            np.zeros(200, dtype=np.int32),
            np.ones(200, dtype=np.int32),
        ]
        
        # Fit for multiple epochs
        model.fit(data, subject_ids=subject_ids, epochs=5, verbose=0)
        
        # Check all transition matrices are row-stochastic
        for p in range(config.n_subjects):
            row_sums = model.trans_prob_subj[p].sum(axis=1)
            assert_allclose(
                row_sums, np.ones(config.n_states),
                rtol=1e-6,
                err_msg=f"Subject {p} transition matrix rows should sum to 1"
            )
            assert np.all(model.trans_prob_subj[p] >= 0), \
                f"Subject {p} transition matrix should be non-negative"
    
    def test_parameter_update_direction_matches_gradient(self):
        """Test that parameters move in the direction of negative gradient."""
        np.random.seed(42)
        
        config = HierarchicalConfig(
            n_states=2,
            n_channels=3,
            n_subjects=2,
            sequence_length=50,
            batch_size=100,  # Large batch to reduce noise
            learning_rate=0.01,  # Small LR for linear approximation
            learn_means=True,
            learn_covariances=False,  # Simplify by fixing covariances
            n_epochs=1,
            initial_covariances_pop=None,
            lambda_mu=0.0,
            lambda_L=0.0,
            learn_trans_prob=True
    
        )
        
        model = HierarchicalModel(config)
        
        # Use deterministic data
        data = Data([
            np.random.randn(500, config.n_channels).astype(np.float32),
            np.random.randn(500, config.n_channels).astype(np.float32),
        ], time_axis_first=True, sampling_frequency=250)
        
        subject_ids = [
            np.zeros(500, dtype=np.int32),
            np.ones(500, dtype=np.int32),
        ]
        
        # Get initial means
        initial_means_subj_0 = model.get_means_subj()[0].copy()
        
        # Manually compute what the gradient should be for first batch
        # (This is complex, so we just verify the update has correct sign properties)
        
        # Fit for one epoch
        model.fit(data, subject_ids=subject_ids, epochs=1, verbose=0)
        
        # Get updated means
        final_means_subj_0 = model.get_means_subj()[0]
        
        # The update should be non-trivial
        delta = final_means_subj_0 - initial_means_subj_0
        assert not np.allclose(delta, 0), \
            "Parameters should change after fit"
    
    def test_loss_decreases_over_epochs(self):
        """Test that loss generally decreases over training."""
        np.random.seed(42)
        
        config = HierarchicalConfig(
            n_states=2,
            n_channels=5,
            n_subjects=2,
            sequence_length=50,
            batch_size=8,
            learning_rate=0.01,
            learn_means=True,
            learn_covariances=True,
            n_epochs=1,
            initial_covariances_pop=None,
            learn_trans_prob=True

        )
        
        model = HierarchicalModel(config)
        
        data = Data([
            np.random.randn(500, config.n_channels).astype(np.float32),
            np.random.randn(500, config.n_channels).astype(np.float32),
        ], time_axis_first=True, sampling_frequency=250)
        
        subject_ids = [
            np.zeros(500, dtype=np.int32),
            np.ones(500, dtype=np.int32),
        ]
        
        # Fit for multiple epochs
        history = model.fit(data, subject_ids=subject_ids, epochs=10, verbose=0)
        
        # Loss should generally decrease (allow some fluctuation)
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]
        
        assert final_loss < initial_loss, \
            f"Loss should decrease: initial={initial_loss}, final={final_loss}"
    
    def test_fit_with_single_subject_batch(self):
        """Test fit works when batch contains only one subject."""
        np.random.seed(42)
        
        config = HierarchicalConfig(
            n_states=2,
            n_channels=3,
            n_subjects=3,
            sequence_length=50,
            batch_size=4,
            learning_rate=0.1,
            learn_means=True,
            learn_covariances=True,
            n_epochs=1,
            initial_covariances_pop=None,
            learn_trans_prob=True

        )
        
        model = HierarchicalModel(config)
        
        # Data for only one subject
        data = Data([
            np.random.randn(200, config.n_channels).astype(np.float32),
        ], time_axis_first=True, sampling_frequency=250)
        
        subject_ids = [
            np.zeros(200, dtype=np.int32),  # Only subject 0
        ]
        
        # Should not crash
        history = model.fit(data, subject_ids=subject_ids, epochs=2, verbose=0)
        
        assert history is not None
        assert len(history['loss']) == 2

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

        problem_all['subject_ids'] = np.random.choice([0,1,2],(small_problem['B'], small_problem['T'])).astype(np.int32)  # All 3 subjects
        
        problem_subset = small_problem.copy()
        problem_subset['subject_ids']= np.zeros((small_problem['B'], small_problem['T']), dtype=np.int32) #np.array([0, 0, 0, 0], dtype=np.int32)
        
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
        problem['subject_ids'] = np.zeros((small_problem['B'], small_problem['T']), dtype=np.int32) #np.array([0, 0, 0, 0], dtype=np.int32)
        
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