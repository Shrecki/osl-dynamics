import logging
import os
import os.path as op
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path


from osl_dynamics.models.hmm import Config, Model
from dataclasses import dataclass
from tensorflow.keras import backend, layers, utils
from osl_dynamics.inference.layers import (
    CategoricalLogLikelihoodLossLayer,
    CovarianceMatricesLayer,
    DiagonalMatricesLayer,
    VectorsLayer,
    StaticLossScalingFactorLayer,
    CholeskyFactorsLayer
)

import numba
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from numba.core.errors import NumbaWarning
from scipy.special import logsumexp, xlogy
from tqdm.auto import trange
from pqdm.threads import pqdm
import logging


import osl_dynamics.data.tf as dtf
from osl_dynamics.inference import initializers, modes

from osl_dynamics.models import obs_mod
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.simulation import HMM
from osl_dynamics.utils.misc import set_logging_level
from osl_dynamics import array_ops
from osl_dynamics.data import Data


from . import _hmmc

_logger = logging.getLogger("osl-dynamics")

warnings.filterwarnings("ignore", category=NumbaWarning)
logging.getLogger("numba.core.transforms").setLevel(logging.ERROR)

EPS = sys.float_info.epsilon

import tensorflow as tf
@dataclass
class HierarchicalConfig(BaseModelConfig):
    model_name: str = "HierarchicalHMM"
    n_subjects: int = None
    lambda_mu: float = 1.0
    lambda_L: float = 1.0
    initial_covariances_pop: np.ndarray = None
    initial_means_pop: np.ndarray = None
    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means_subj: np.ndarray = None
    initial_covariances_subj: np.ndarray = None
    diagonal_covariances: bool = False
    covariances_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

    initial_trans_prob_subj: np.ndarray = None
    learn_trans_prob: bool = True
    state_probs_t0_subj: np.ndarray = None

    # Learning rate schedule parameters
    trans_prob_update_delay: float = 5  # alpha
    trans_prob_update_forget: float = 0.7  # beta
    observation_update_decay: float = 0.1
    
    # Transition model parameters
    #kappa: float = 0.0
    
    # Use log space or not
    implementation: str = "log"
    
    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_trans_prob_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

        if self.covariances_epsilon is None:
            if self.learn_covariances:
                self.covariances_epsilon = 1e-6
            else:
                self.covariances_epsilon = 0.0
        
        if self.initial_covariances_pop is not None:
            K = self.n_states

            covs = self.initial_covariances_pop
            if covs.shape[0] != K:
                raise ValueError("initial_covariances_pop should be n_states x D x D")
        
        if self.initial_covariances_subj is not None:
            K = self.n_states

            covs = self.initial_covariances_subj
            if covs.shape[0] != K:
                raise ValueError("initial_covariances_subj should be n_states x D x D")
        else: 
            self.initial_covariances_subj = [None]*self.n_subjects
            
            
        if self.initial_means_pop is not None:
            K = self.n_states

            covs = self.initial_means_pop
            if covs.shape[0] != K:
                raise ValueError("initial_means_pop should be n_states x D")
        
        if self.initial_means_subj is not None:
            K = self.n_states

            covs = self.initial_means_subj
            if covs.shape[0] != K:
                raise ValueError("initial_means_subj should be n_states x D")
        else: 
            self.initial_means_subj = [None]*self.n_subjects

    def validate_trans_prob_parameters(self):
        if self.initial_trans_prob_subj is not None:
            P = self.initial_trans_prob_subj
            ns = self.n_subjects
            K = self.n_states
            if (not isinstance(P, np.ndarray) or P.ndim != 3 or not(np.all(P.shape == (ns,K,K)))):
                raise ValueError("initial_trans_prob_subj must be a 3D numpy array of shape n_subjects x n_states x n_states")
            for s in range(ns):
                if not all(np.isclose(np.sum(P[s], axis=1), 1)):
                    raise ValueError(f"rows of initial_trans_prob_subj must sum to one, but failed for subj {s}")
    
    
    def validate_hierarchical_parameters(self):
        if self.n_subjects is None:
            raise ValueError("n_subjects must be passed. Otherwise use regular config and HMM")
        if self.n_subjects < 1:
            raise ValueError("n_subjects must be one or greater")
        if self.lambda_mu < 0:
            raise ValueError("lambda_mu must be non-negative")
        if self.lambda_L < 0:
            raise ValueError("lambda_L must be non-negative")
        

"""Hierarchical Categorical Log-Likelihood Layer with Analytical Gradients.

This module implements the hierarchical HMM loss function with custom gradients
that correctly handle:
- Subject-specific parameters (only updated when their data is in batch)
- Population parameters (updated based on present subjects, scaled appropriately)

The loss function is:
    L = -1/2 * sum_{p,t,i} gamma_i^p(t) * [||L_i^p^{-1}(x_t^p - mu_i^p)||_2^2 + 2*sum_d log(L_i^p)_dd]
        - lambda_mu/2 * sum_{p,i} ||mu_i^p - mu_i^pop||_2^2
        - lambda_L/2 * sum_{p,i} ||L_i^p - L_i^pop||_F^2
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def sort_and_rowsplit_by_subject(x, gamma, subject_ids):
    """
    Flatten (B,T,...) -> (N,...), sort by subject id, and compute row_splits for subject blocks.

    Returns
    -------
    unique_subjects : (S,) int32   subjects present in batch, in sorted order
    row_splits      : (S+1,) int32 ragged row splits into the sorted flattened arrays
    x_sorted        : (N,D)
    gamma_sorted    : (N,K)
    sid_sorted      : (N,)
    """
    x = tf.convert_to_tensor(x)
    gamma = tf.convert_to_tensor(gamma)
    subject_ids = tf.cast(subject_ids, tf.int32)
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    D = tf.shape(x)[2]
    K = tf.shape(gamma)[2]
    N = B * T
    # Flatten
    x_flat = tf.reshape(x, [N, D])
    g_flat = tf.reshape(gamma, [N, K])
    sid_flat = tf.reshape(subject_ids, [N])
    # Sort by subject id
    sort_idx = tf.argsort(sid_flat, stable=True)
    sid_sorted = tf.gather(sid_flat, sort_idx)
    x_sorted = tf.gather(x_flat, sort_idx)
    gamma_sorted = tf.gather(g_flat, sort_idx)
    # Find boundaries where subject id changes
    change = tf.not_equal(sid_sorted[1:], sid_sorted[:-1])          # (N-1,)
    boundaries = tf.where(change)[:, 0] + 1                         # positions in 1..N-1
    row_splits = tf.concat(
        [
            tf.zeros([1], dtype=tf.int32),
            tf.cast(boundaries, tf.int32),
            tf.reshape(tf.cast(N, tf.int32), [1])
        ],
        axis=0
    )
    # Unique subjects in the same order as blocks
    unique_subjects = tf.gather(sid_sorted, row_splits[:-1])
    return unique_subjects, row_splits, x_sorted, gamma_sorted, sid_sorted

@tf.custom_gradient
def hierarchical_categorical_nll_mean_custom_grad(
    x, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids, lambda_mu, lambda_L, n_subjects
):
    """
    Hierarchical categorical NLL with analytical gradients.
    
    Parameters
    ----------
    x : tf.Tensor
        Data, shape (B, T, D) where B=batch, T=time, D=channels.
    mu_pop : tf.Tensor
        Population means, shape (K, D) where K=states.
    L_pop : tf.Tensor
        Population Cholesky factors, shape (K, D, D).
    mu_subj : tf.Tensor
        Subject means, shape (P, K, D) where P=n_subjects.
    L_subj : tf.Tensor
        Subject Cholesky factors, shape (P, K, D, D).
    gamma : tf.Tensor
        State responsibilities, shape (B, T, K).
    subject_ids : tf.Tensor
        Subject ID for each batch element, shape (B,).
    lambda_mu : float
        Regularization strength for means.
    lambda_L : float
        Regularization strength for Cholesky factors.
    n_subjects : int
        Total number of subjects P.
    
    Returns
    -------
    total_loss : tf.Tensor
        Scalar loss value.
    """
    dtype = mu_pop.dtype
    x = tf.cast(x, dtype)
    gamma = tf.cast(gamma, dtype)
    lambda_mu = tf.cast(lambda_mu, dtype)
    lambda_L = tf.cast(lambda_L, dtype)
    n_subjects = tf.cast(n_subjects, dtype)
    
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    D = tf.shape(x)[2]
    K = tf.shape(mu_pop)[0]
    P = tf.shape(mu_subj)[0]
    subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)

    unique_subjects, row_splits, x_sorted, gamma_sorted, _ = sort_and_rowsplit_by_subject(
        x, gamma, subject_ids
    )

    N = tf.cast(B*T, dtype)
    S = tf.shape(unique_subjects)[0]
    
    # =====================
    # Forward pass: compute NLL and cache intermediates for backward
    # =====================
    
    def per_subject_forward(i):
        start = row_splits[i]
        stop  = row_splits[i + 1]
        sid   = unique_subjects[i]

        x_block = x_sorted[start:stop]         # (N_i, D)
        g_block = gamma_sorted[start:stop]     # (N_i, K)

        mu_i = tf.gather(mu_subj, sid, axis=0) # (K, D)
        L_i  = tf.gather(L_subj,  sid, axis=0) # (K, D, D)

        # Residuals: (N_i, K, D)
        r = x_block[:, None, :] - mu_i[None, :, :]
        
        # y = L^{-1} r: (N_i, K, D)
        y = tf.linalg.triangular_solve(L_i[None, ...], r[..., None], lower=True)
        y = tf.squeeze(y, axis=-1)
        
        # Mahalanobis squared: (N_i, K)
        mahal_sq = tf.reduce_sum(tf.square(y), axis=-1)
        
        # Log det: (K,)
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_i)), axis=-1)
        
        # Log prob: (N_i, K)
        log_2pi = tf.cast(tf.math.log(2.0 * np.pi), dtype)
        D_float = tf.cast(D, dtype)
        logp = -0.5 * (D_float * log_2pi + log_det[None, :] + mahal_sq)
        
        # NLL contribution (scalar)
        nll_contrib = -tf.reduce_sum(g_block * logp)
        
        # Cache for backward: weighted sums we'll need
        w = g_block / N  # (N_i, K)
        
        # For mu gradient: need sum of w * u where u = L^{-T} y
        u = tf.linalg.triangular_solve(
            tf.linalg.matrix_transpose(L_i)[None, ...],
            y[..., None],
            lower=False
        )
        u = tf.squeeze(u, axis=-1)  # (N_i, K, D)
        sum_wu = tf.einsum("nk,nkd->kd", w, u)  # (K, D)
        
        # For L gradient: need sum of w * r * r^T -> (K, D, D)
        S_k = tf.einsum("nk,nkd,nkf->kdf", w, r, r)
        
        # N_k = sum of weights per state: (K,)
        N_k = tf.reduce_sum(w, axis=0)
        
        return nll_contrib, sid, sum_wu, S_k, N_k

    # Run forward pass for all subjects
    nll_contribs, sids, sum_wus, S_ks, N_ks = tf.map_fn(
        per_subject_forward,
        tf.range(S),
        fn_output_signature=(
            tf.TensorSpec(shape=(), dtype=dtype),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=dtype),        # (K, D)
            tf.TensorSpec(shape=(None, None, None), dtype=dtype),  # (K, D, D)
            tf.TensorSpec(shape=(None,), dtype=dtype),             # (K,)
        )
    )
    
    nll = tf.reduce_sum(nll_contribs) / N
    
    # =====================
    # Regularization loss (unchanged, already efficient)
    # =====================
    
    mu_subj_present = tf.gather(mu_subj, unique_subjects, axis=0)
    L_subj_present = tf.gather(L_subj, unique_subjects, axis=0)
    
    mu_diff = mu_subj_present - mu_pop[None, :, :]
    L_diff = L_subj_present - L_pop[None, :, :, :]
    
    total_mu_reg = tf.reduce_sum(tf.square(mu_diff))
    total_L_reg = tf.reduce_sum(tf.square(L_diff))
    
    scale = tf.cast(S, dtype) / n_subjects
    reg_loss = scale * (lambda_mu / 2.0 * total_mu_reg + lambda_L / 2.0 * total_L_reg)
    
    total_loss = nll + reg_loss
    
    # =====================
    # Custom gradient using cached values
    # =====================
    
    def grad(dy):
        dy = tf.cast(dy, dtype)
        
        I = tf.eye(D, dtype=dtype)
        
        def compute_grad_for_subject(i):
            sid = sids[i]
            sum_wu_i = sum_wus[i]  # (K, D) - cached
            S_k_i = S_ks[i]        # (K, D, D) - cached
            N_k_i = N_ks[i]        # (K,) - cached
            
            L_i = tf.gather(L_subj, sid, axis=0)  # (K, D, D)
            
            # mu gradient: -sum_wu
            gmu = -sum_wu_i  # (K, D)
            
            # L gradient using cached S_k and N_k
            LinvT = tf.linalg.triangular_solve(
                tf.linalg.matrix_transpose(L_i),
                I[None, :, :],
                lower=False
            )  # (K, D, D)
            
            tmp = tf.linalg.triangular_solve(L_i, S_k_i, lower=True)
            A = tf.matmul(tmp, LinvT)  # (K, D, D)
            
            NI_minus_A = N_k_i[:, None, None] * I[None, :, :] - A
            gL = tf.matmul(LinvT, NI_minus_A)
            gL = tf.linalg.band_part(gL, -1, 0)  # lower tri
            
            return sid, gmu, gL

        sids_out, gmus, gLs = tf.map_fn(
            compute_grad_for_subject,
            tf.range(S),
            fn_output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=dtype),
                tf.TensorSpec(shape=(None, None, None), dtype=dtype)
            )
        )

        # Scatter gradients
        grad_mu_subj_nll = tf.zeros_like(mu_subj)
        grad_L_subj_nll = tf.zeros_like(L_subj)
        
        indices = tf.expand_dims(sids_out, 1)
        grad_mu_subj_nll = tf.tensor_scatter_nd_add(grad_mu_subj_nll, indices, gmus)
        grad_L_subj_nll = tf.tensor_scatter_nd_add(grad_L_subj_nll, indices, gLs)

        # Regularization gradients
        subject_mask = tf.scatter_nd(
            tf.expand_dims(unique_subjects, 1),
            tf.ones_like(unique_subjects, dtype=dtype),
            [P]
        )

        reg_scale = scale
        
        grad_mu_subj_reg = lambda_mu * (mu_subj - mu_pop[None, :, :])
        grad_mu_subj_reg *= subject_mask[:, None, None] * reg_scale

        grad_L_subj_reg = lambda_L * (L_subj - L_pop[None, :, :, :])
        grad_L_subj_reg *= subject_mask[:, None, None, None] * reg_scale

        grad_mu_subj_total = dy * (grad_mu_subj_nll + grad_mu_subj_reg)
        grad_L_subj_total = dy * (grad_L_subj_nll + grad_L_subj_reg)

        mu_diff_masked = (mu_subj - mu_pop[None, :, :]) * subject_mask[:, None, None]
        grad_mu_pop = -dy * lambda_mu * tf.reduce_sum(mu_diff_masked, axis=0) * reg_scale

        L_diff_masked = (L_subj - L_pop[None, :, :, :]) * subject_mask[:, None, None, None]
        grad_L_pop = -dy * lambda_L * tf.reduce_sum(L_diff_masked, axis=0) * reg_scale

        return (None, grad_mu_pop, grad_L_pop, grad_mu_subj_total, grad_L_subj_total, 
                None, None, None, None, None)
    
    return total_loss, grad


@tf.custom_gradient
def hierarchical_categorical_nll_sum_custom_grad(
    x, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids, lambda_mu, lambda_L, n_subjects
):
    dtype = mu_pop.dtype
    x = tf.cast(x, dtype)
    gamma = tf.cast(gamma, dtype)
    lambda_mu = tf.cast(lambda_mu, dtype)
    lambda_L = tf.cast(lambda_L, dtype)
    n_subjects = tf.cast(n_subjects, dtype)
    
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    D = tf.shape(x)[2]
    K = tf.shape(mu_pop)[0]
    P = tf.shape(mu_subj)[0]
    
    subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
    
    unique_subjects, row_splits, x_sorted, gamma_sorted, _ = sort_and_rowsplit_by_subject(
        x, gamma, subject_ids
    )

    N = tf.cast(B, dtype)
    S = tf.shape(unique_subjects)[0]
    
    # =====================
    # Forward pass: compute NLL and cache intermediates for backward
    # =====================
    
    def per_subject_forward(i):
        start = row_splits[i]
        stop  = row_splits[i + 1]
        sid   = unique_subjects[i]

        x_block = x_sorted[start:stop]         # (N_i, D)
        g_block = gamma_sorted[start:stop]     # (N_i, K)

        mu_i = tf.gather(mu_subj, sid, axis=0) # (K, D)
        L_i  = tf.gather(L_subj,  sid, axis=0) # (K, D, D)

        # Residuals: (N_i, K, D)
        r = x_block[:, None, :] - mu_i[None, :, :]
        
        # y = L^{-1} r: (N_i, K, D)
        y = tf.linalg.triangular_solve(L_i[None, ...], r[..., None], lower=True)
        y = tf.squeeze(y, axis=-1)
        
        # Mahalanobis squared: (N_i, K)
        mahal_sq = tf.reduce_sum(tf.square(y), axis=-1)
        
        # Log det: (K,)
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_i)), axis=-1)
        
        # Log prob: (N_i, K)
        log_2pi = tf.cast(tf.math.log(2.0 * np.pi), dtype)
        D_float = tf.cast(D, dtype)
        logp = -0.5 * (D_float * log_2pi + log_det[None, :] + mahal_sq)
        
        # NLL contribution (scalar)
        nll_contrib = -tf.reduce_sum(g_block * logp)
        
        # Cache for backward: weighted sums we'll need
        w = g_block / N  # (N_i, K)
        
        # For mu gradient: need sum of w * u where u = L^{-T} y
        u = tf.linalg.triangular_solve(
            tf.linalg.matrix_transpose(L_i)[None, ...],
            y[..., None],
            lower=False
        )
        u = tf.squeeze(u, axis=-1)  # (N_i, K, D)
        sum_wu = tf.einsum("nk,nkd->kd", w, u)  # (K, D)
        
        # For L gradient: need sum of w * r * r^T -> (K, D, D)
        S_k = tf.einsum("nk,nkd,nkf->kdf", w, r, r)
        
        # N_k = sum of weights per state: (K,)
        N_k = tf.reduce_sum(w, axis=0)
        
        return nll_contrib, sid, sum_wu, S_k, N_k

    # Run forward pass for all subjects
    nll_contribs, sids, sum_wus, S_ks, N_ks = tf.map_fn(
        per_subject_forward,
        tf.range(S),
        fn_output_signature=(
            tf.TensorSpec(shape=(), dtype=dtype),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=dtype),        # (K, D)
            tf.TensorSpec(shape=(None, None, None), dtype=dtype),  # (K, D, D)
            tf.TensorSpec(shape=(None,), dtype=dtype),             # (K,)
        )
    )
    
    nll = tf.reduce_sum(nll_contribs) / N
    
    # =====================
    # Regularization loss (unchanged, already efficient)
    # =====================
    
    mu_subj_present = tf.gather(mu_subj, unique_subjects, axis=0)
    L_subj_present = tf.gather(L_subj, unique_subjects, axis=0)
    
    mu_diff = mu_subj_present - mu_pop[None, :, :]
    L_diff = L_subj_present - L_pop[None, :, :, :]
    
    total_mu_reg = tf.reduce_sum(tf.square(mu_diff))
    total_L_reg = tf.reduce_sum(tf.square(L_diff))
    
    scale = tf.cast(S, dtype) / n_subjects
    reg_loss = scale * (lambda_mu / 2.0 * total_mu_reg + lambda_L / 2.0 * total_L_reg)
    
    total_loss = nll + reg_loss
    
    # =====================
    # Custom gradient using cached values
    # =====================
    
    def grad(dy):
        dy = tf.cast(dy, dtype)
        
        I = tf.eye(D, dtype=dtype)
        
        def compute_grad_for_subject(i):
            sid = sids[i]
            sum_wu_i = sum_wus[i]  # (K, D) - cached
            S_k_i = S_ks[i]        # (K, D, D) - cached
            N_k_i = N_ks[i]        # (K,) - cached
            
            L_i = tf.gather(L_subj, sid, axis=0)  # (K, D, D)
            
            # mu gradient: -sum_wu
            gmu = -sum_wu_i  # (K, D)
            
            # L gradient using cached S_k and N_k
            LinvT = tf.linalg.triangular_solve(
                tf.linalg.matrix_transpose(L_i),
                I[None, :, :],
                lower=False
            )  # (K, D, D)
            
            tmp = tf.linalg.triangular_solve(L_i, S_k_i, lower=True)
            A = tf.matmul(tmp, LinvT)  # (K, D, D)
            
            NI_minus_A = N_k_i[:, None, None] * I[None, :, :] - A
            gL = tf.matmul(LinvT, NI_minus_A)
            gL = tf.linalg.band_part(gL, -1, 0)  # lower tri
            
            return sid, gmu, gL

        sids_out, gmus, gLs = tf.map_fn(
            compute_grad_for_subject,
            tf.range(S),
            fn_output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=dtype),
                tf.TensorSpec(shape=(None, None, None), dtype=dtype)
            )
        )

        # Scatter gradients
        grad_mu_subj_nll = tf.zeros_like(mu_subj)
        grad_L_subj_nll = tf.zeros_like(L_subj)
        
        indices = tf.expand_dims(sids_out, 1)
        grad_mu_subj_nll = tf.tensor_scatter_nd_add(grad_mu_subj_nll, indices, gmus)
        grad_L_subj_nll = tf.tensor_scatter_nd_add(grad_L_subj_nll, indices, gLs)

        # Regularization gradients
        subject_mask = tf.scatter_nd(
            tf.expand_dims(unique_subjects, 1),
            tf.ones_like(unique_subjects, dtype=dtype),
            [P]
        )

        reg_scale = scale
        
        grad_mu_subj_reg = lambda_mu * (mu_subj - mu_pop[None, :, :])
        grad_mu_subj_reg *= subject_mask[:, None, None] * reg_scale

        grad_L_subj_reg = lambda_L * (L_subj - L_pop[None, :, :, :])
        grad_L_subj_reg *= subject_mask[:, None, None, None] * reg_scale

        grad_mu_subj_total = dy * (grad_mu_subj_nll + grad_mu_subj_reg)
        grad_L_subj_total = dy * (grad_L_subj_nll + grad_L_subj_reg)

        mu_diff_masked = (mu_subj - mu_pop[None, :, :]) * subject_mask[:, None, None]
        grad_mu_pop = -dy * lambda_mu * tf.reduce_sum(mu_diff_masked, axis=0) * reg_scale

        L_diff_masked = (L_subj - L_pop[None, :, :, :]) * subject_mask[:, None, None, None]
        grad_L_pop = -dy * lambda_L * tf.reduce_sum(L_diff_masked, axis=0) * reg_scale

        return (None, grad_mu_pop, grad_L_pop, grad_mu_subj_total, grad_L_subj_total, 
                None, None, None, None, None)
    
    return total_loss, grad


class HierarchicalCategoricalLogLikelihoodLossLayer(tf.keras.layers.Layer):
    """Layer to calculate hierarchical log-likelihood loss with analytical gradients.
    
    Parameters
    ----------
    n_states : int
        Number of HMM states.
    n_subjects : int
        Total number of subjects.
    epsilon : float
        Error added to covariances for numerical stability.
    calculation : str
        'mean' or 'sum' for time dimension reduction.
    lambda_mu : float
        Regularization strength for means.
    lambda_L : float
        Regularization strength for Cholesky factors.
    """
    
    
    def __init__(
        self,
        n_states,
        n_subjects,
        epsilon,
        calculation,
        lambda_mu=1.0,
        lambda_L=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.n_subjects = n_subjects
        self.epsilon = epsilon
        self.calculation = calculation
        self.lambda_mu = lambda_mu
        self.lambda_L = lambda_L
    
    def call(self, inputs, **kwargs):
        """
        Parameters
        ----------
        inputs : list
            [data, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids]
        
        Returns
        -------
        loss : tf.Tensor
            Scalar loss.
        """
        data, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids = inputs
        
        if self.calculation == "sum":
            total_loss = hierarchical_categorical_nll_sum_custom_grad(
                data, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids,
                self.lambda_mu, self.lambda_L, self.n_subjects
            )
        else:
            total_loss = hierarchical_categorical_nll_mean_custom_grad(
                data, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids,
                self.lambda_mu, self.lambda_L, self.n_subjects
            )
        
        self.add_loss(total_loss)
        self.add_metric(total_loss, name=self.name)
        
        return tf.expand_dims(total_loss, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_states": self.n_states,
            "n_subjects": self.n_subjects,
            "epsilon": self.epsilon,
            "calculation": self.calculation,
            "lambda_mu": self.lambda_mu,
            "lambda_L": self.lambda_L,
        })
        return config
        
class HierarchicalModel(Model):
    """HMM class.

    Parameters
    ----------
    config : osl_dynamics.models.hmm.Config
    """

    config_type = HierarchicalConfig
    
    def set_trans_prob(self, trans_prob_subj, subject_id=None):
        """Sets the transition probability matrix.

        Parameters
        ----------
        trans_prob : np.ndarray
            State transition probabilities. Shape must be (n_subjects, n_states, n_states).
        """
        if trans_prob_subj is None:
            if subject_id is None:
                trans_prob_subj = (
                    np.ones((self.config.n_subjects, self.config.n_states, self.config.n_states))
                    * 0.1
                    / (self.config.n_states - 1)
                )
                for i in range(self.config.n_subjects):
                    np.fill_diagonal(trans_prob_subj[i], 0.9)
                self.trans_prob_subj = trans_prob_subj

            else:
                self.trans_prob_subj[subject_id] = trans_prob_subj
        
        
    def set_state_probs_t0(self, state_probs_t0_subj):
        """Set the initial state probabilities.

        Parameters
        ----------
        state_probs_t0 : np.ndarray
            Initial state probabilities. Shape is (n_subjects, n_states,).
        """

        if state_probs_t0_subj is None:
            state_probs_t0_subj = np.ones((self.config.n_subjects, self.config.n_states,)) / self.config.n_states
        self.state_probs_t0_subj = state_probs_t0_subj
        
    def build_model(self):
        """Builds a keras model."""
        self.model = self._model_structure()

        self.rho = 1
        self.set_trans_prob(self.config.initial_trans_prob_subj)
        self.set_state_probs_t0(self.config.state_probs_t0_subj)
    
    def __init__(self, config):
        # Population transition probability
        #self.trans_prob_pop = None
        
        # Subject-specific transition probabilities (M matrices)
        #self.trans_prob_subj = [None] * config.n_subjects
        super().__init__(config)
        
        

    def get_trans_prob(self, subject_id=None):
        """Get transition probability matrix."""
        if subject_id is None:
            return self.trans_prob_pop
        else:
            return self.trans_prob_subj[subject_id]
    
    def _model_structure(self):
        config = self.config
        
        # Definition of layers
        data_inputs = layers.Input(
            shape=(config.sequence_length, config.n_channels + config.n_states),
            name="data_inputs",
        )
        subject_id_inputs = layers.Input(
            shape=(1,),  # Subject ID for each sequence in the batch
            dtype=tf.int32,
            name="subject_id_inputs",
        )
        inputs = [data_inputs, subject_id_inputs]
        
        static_loss_scaling_factor_layer = StaticLossScalingFactorLayer(
            config.sequence_length,
            config.loss_calc,
            name="static_loss_scaling_factor",
        )
        
        # Population means
        means_pop_layer = VectorsLayer(
            config.n_states,
            config.n_channels,
            config.learn_means,
            config.initial_means_pop,
            None,  # No regularizer on population params
            name="means_pop",
        )
        
        # Subject-specific means (M subjects)
        means_subj_layers = []
        for m in range(config.n_subjects):
            means_subj_layers.append(
                VectorsLayer(
                    config.n_states,
                    config.n_channels,
                    config.learn_means,
                    config.initial_means_subj[m],
                    None,
                    name=f"means_subj_{m}",
                )
            )
        
        # Population Cholesky factors
        init_L_pop = tf.linalg.cholesky(tf.convert_to_tensor(config.initial_covariances_pop, tf.float32)) if config.initial_covariances_pop is not None else None
        
        trils_pop_layer = CholeskyFactorsLayer(
            config.n_states,
            config.n_channels,
            config.learn_covariances,
            init_L_pop,
            config.covariances_epsilon,
            None,
            name="trils_pop",
        )
        
        # Subject-specific Cholesky factors
        trils_subj_layers = []
        for m in range(config.n_subjects):
            init_L = tf.linalg.cholesky(tf.convert_to_tensor(config.initial_covariances_subj[m], tf.float32)) if config.initial_covariances_subj is not None and config.initial_covariances_subj[m] is not None else None
            trils_subj_layers.append(
                CholeskyFactorsLayer(
                    config.n_states,
                    config.n_channels,
                    config.learn_covariances,
                    init_L,
                    config.covariances_epsilon,
                    None,
                    name=f"trils_subj_{m}",
                )
            )
        
        # Data flow
        data, gamma = tf.split(data_inputs, [config.n_channels, config.n_states], axis=2)
        static_loss_scaling_factor = static_loss_scaling_factor_layer(data)
        
        mu_pop = means_pop_layer(data, static_loss_scaling_factor=static_loss_scaling_factor)
        L_pop = trils_pop_layer(data, static_loss_scaling_factor=static_loss_scaling_factor)
        
        mu_subj_list = []
        L_subj_list = []
        for m in range(config.n_subjects):
            mu_m = means_subj_layers[m](data, static_loss_scaling_factor=static_loss_scaling_factor)
            L_m = trils_subj_layers[m](data, static_loss_scaling_factor=static_loss_scaling_factor)
            mu_subj_list.append(mu_m)
            L_subj_list.append(L_m)
        
        mu_subj = tf.stack(mu_subj_list, axis=0)  # (P, K, D)
        L_subj = tf.stack(L_subj_list, axis=0)    # (P, K, D, D)
        
        # Hierarchical loss layer
        ll_loss_layer = HierarchicalCategoricalLogLikelihoodLossLayer(
            n_states=config.n_states,
            n_subjects=config.n_subjects,
            epsilon=config.covariances_epsilon,
            calculation=config.loss_calc,
            lambda_mu=config.lambda_mu,
            lambda_L=config.lambda_L,
            name="ll_loss",
        )
        
        # NOTE: inputs order must match what the layer expects
        ll_loss = ll_loss_layer([data, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_id_inputs])
        
        model = tf.keras.Model(inputs=inputs, outputs=[ll_loss], name="HierarchicalHMM")
        return model
        
    def get_observation_model_parameter_tensor(self, layer_name):
        """Get the parameter of an observation model layer, in tensor form.

        Parameters
        ----------
        model : osl_dynamics.models.*.Model.model
            The model.
        layer_name : str
            Name of the layer of the observation model parameter.

        Returns
        -------
        obs_parameter : tf.tensor
            The observation model parameter.
        """
        available_layers = [
            "means",
            "covs",
            "stds",
            "corrs",
            "group_means",
            "group_covs",
            "log_rates",
            "trils"
        ]
        isin = np.array([(layer_name in layer) or (layer in layer_name) for layer in available_layers])
        if not np.any(isin):
            raise ValueError(
                f"Layer name {layer_name} not in available layers {available_layers}."
            )
        obs_layer = self.model.get_layer(layer_name)
        obs_parameter = obs_layer(1)
        return obs_parameter

    def set_observation_model_parameter(
        self,
        obs_parameter,
        layer_name,
        update_initializer=True,
        diagonal_covariances=False,
        ):
        """Set the value of an observation model parameter.

        Parameters
        ----------
        model : osl_dynamics.models.*.Model.model
            The model.
        obs_parameter : np.ndarray
            The value of the observation model parameter to set.
        layer_name : str
            Layer name of the observation model parameter.
        update_initializer : bool, optional
            Whether to update the initializer of the layer.
        diagonal_covariances : bool, optional
            Whether the covariances are diagonal.
            Ignored if :code:`layer_name` is not :code:`"covs"`.
        """
        available_layers = [
            "means",
            "covs",
            "stds",
            "corrs",
            "group_means",
            "group_covs",
            "log_rates",
            "trils"
        ]
        from osl_dynamics.inference.initializers import (
            WeightInitializer,
            RandomWeightInitializer,
            )
        isin = np.array([(layer_name in layer) or (layer in layer_name) for layer in available_layers])
        if not np.any(isin):
            raise ValueError(
                f"Layer name {layer_name} not in available layers {available_layers}."
            )
        if layer_name != "trils":
            obs_parameter = obs_parameter.astype(np.float32)

        if layer_name == "stds" or (layer_name == "covs" and diagonal_covariances):
            if obs_parameter.ndim == 3:
                # Only keep the diagonal as a vector
                obs_parameter = np.diagonal(obs_parameter, axis1=1, axis2=2)

        obs_layer = self.model.get_layer(layer_name)
        learnable_tensor_layer = obs_layer.layers[0]

        if layer_name not in ["means", "group_means", "log_rates"]:
            #print(obs_parameter.shape)
            obs_parameter = obs_layer.bijector.inverse(obs_parameter)

        learnable_tensor_layer.tensor.assign(obs_parameter)

        if update_initializer:
            learnable_tensor_layer.tensor_initializer = WeightInitializer(obs_parameter)

    
    def get_means_subj(self, tensor=False):
        """Get the state means.

        Returns
        -------
        means : np.ndarray
            State means. Shape is (n_states, n_channels).
        """
        if tensor:
            return tf.stack([self.get_observation_model_parameter_tensor(f"means_subj_{m}") for m in range(self.config.n_subjects)],axis=0)
        else:
            return [self.get_observation_model_parameter(f"means_subj_{m}") for m in range(self.config.n_subjects)]
    
    
    def get_trils_subj(self, tensor=False):
        if tensor:
            return tf.stack([self.get_observation_model_parameter_tensor(f"trils_subj_{m}") for m in range(self.config.n_subjects)],axis=0)
        else:    
            return [self.get_observation_model_parameter(f"trils_subj_{m}") for m in range(self.config.n_subjects)]
        
    def get_means_pop(self, tensor=False):
        """Get the state means.

        Returns
        -------
        means : np.ndarray
            State means. Shape is (n_states, n_channels).
        """
        if tensor:
            return self.get_observation_model_parameter_tensor("means_pop")
        else:
            return self.get_observation_model_parameter("means_pop")
    
    def get_trils_pop(self, tensor=False):
        if tensor:
            return self.get_observation_model_parameter_tensor("trils_pop")
        else:    
            return self.get_observation_model_parameter("trils_pop")
        
    
    def get_log_likelihood_graph_hierarchical(self, means_subj, means_pop, trils_subj, trils_pop, data):
        pass
    
    def sort_and_rowsplit_by_subject(self, x, gamma_or_none, subject_ids):
        """
        x: (B,T,D)
        subject_ids: (B,T) int
        Returns:
        unique_subjects: (S,)
        row_splits: (S+1,)
        x_sorted: (N,D)
        (optional) gamma_sorted: (N,K) or None
        perm: (N,) permutation indices s.t. x_sorted = x_flat[perm]
        inv_perm: (N,) inverse permutation s.t. x_flat = x_sorted[inv_perm]
        """
        x = tf.convert_to_tensor(x)
        subject_ids = tf.convert_to_tensor(subject_ids, tf.int32)

        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        D = tf.shape(x)[2]
        N = B * T

        x_flat = tf.reshape(x, [N, D])
        sid_flat = tf.reshape(subject_ids, [N])

        perm = tf.argsort(sid_flat, stable=True)
        sid_sorted = tf.gather(sid_flat, perm)
        x_sorted = tf.gather(x_flat, perm)

        if gamma_or_none is not None:
            g = tf.reshape(gamma_or_none, [N, tf.shape(gamma_or_none)[-1]])
            gamma_sorted = tf.gather(g, perm)
        else:
            gamma_sorted = None

        # segment boundaries
        unique_subjects, first_idx = tf.unique(sid_sorted)  # NOTE: tf.unique keeps first occurrence order
        # But sid_sorted is sorted, so unique_subjects will also be sorted. Good.

        # Build row_splits from counts
        # counts per unique id
        # (fast way) use tf.math.bincount with minlength=max+1 then gather,
        # but P might be large; simplest:
        is_new = tf.concat([[True], sid_sorted[1:] != sid_sorted[:-1]], axis=0)
        boundaries = tf.where(is_new)[:, 0]  # indices where new segment starts
        # boundaries includes 0
        row_splits = tf.concat([boundaries, [N]], axis=0)  # (S+1,)

        # unique_subjects should match sid_sorted[row_splits[:-1]]
        unique_subjects = tf.gather(sid_sorted, row_splits[:-1])

        inv_perm = tf.argsort(perm, stable=True)
        return unique_subjects, row_splits, x_sorted, gamma_sorted, perm, inv_perm


    def hierarchical_logB(self,
        x, subject_ids, mu_subj, L_subj
    ):
        """
        x: (B,T,D) (or (N,D) if you adapt)
        subject_ids: (B,T)
        mu_subj: (P,K,D)
        L_subj: (P,K,D,D)
        returns logB: (B,T,K)
        """
        dtype = mu_subj.dtype
        x = tf.cast(x, dtype)
        subject_ids = tf.cast(subject_ids, tf.int32)

        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        D = tf.shape(x)[2]
        K = tf.shape(mu_subj)[1]
        N = B * T

        unique_subjects, row_splits, x_sorted, _, perm, inv_perm = self.sort_and_rowsplit_by_subject(
            x, None, subject_ids
        )
        S = tf.shape(unique_subjects)[0]

        # compute logB in sorted order, per subject block
        logB_sorted =  tf.TensorArray(
                        dtype=dtype,
                        size=S,
                        infer_shape=False,                    # key
                        element_shape=tf.TensorShape([None, None])  # (N_i, K) where N_i varies
                    )

        for i in tf.range(S):
            start = row_splits[i]
            stop  = row_splits[i + 1]
            sid = unique_subjects[i]

            x_block = x_sorted[start:stop]           # (N_i, D)
            mu_i = tf.gather(mu_subj, sid, axis=0)   # (K, D)
            L_i  = tf.gather(L_subj,  sid, axis=0)   # (K, D, D)

            mvn = tfd.MultivariateNormalTriL(loc=mu_i, scale_tril=L_i, allow_nan_stats=False)

            # broadcast: x_block[:,None,:] is (N_i,1,D) -> log_prob (N_i,K)
            logp = mvn.log_prob(x_block[:, None, :])  # (N_i, K)
            
            logB_sorted = logB_sorted.write(i, logp)

        logB_sorted = logB_sorted.concat() #tf.concat(logB_sorted.stack(), axis=0)  # (N,K) in sorted order

        # unsort back to original flat order
        logB_flat = tf.gather(logB_sorted, inv_perm)          # (N,K)
        logB = tf.reshape(logB_flat, [B, T, K])  # (B,T,K)
        return logB

    
    def get_log_likelihood_hierarchical(self, data, subject_ids):
        r"""Get the log-likelihood of data, :math:`\log p(x_t | s_t)`.

        Parameters
        ----------
        data : np.ndarray
            Data. Shape is (batch_size, ..., n_channels).

        Returns
        -------
        log_likelihood : np.ndarray
            Log-likelihood. Shape is (batch_size, ..., n_states)"""
        #data = tf.convert_to_tensor(data, dtype=tf.float32)
        assert tf.is_tensor(data)
        
        # Get all parameters (population + subjects)
        means_subj = self.get_means_subj(tensor=True)
        
        L_subj = self.get_trils_subj(tensor=True)
        
        return  self.hierarchical_logB(data, subject_ids[:,:,0], means_subj, L_subj).numpy()
    
    def get_posterior(self, x, subj_ids):
        """Get marginal and joint posterior.

        Parameters
        ----------
        x : np.ndarray
            Observed data. Shape is (batch_size, sequence_length, n_channels).
        subj_ids: np.ndarray
            IDs of subjects, indicating which are present for hierarchical modeling. Shape is (batch_size, sequence_length,1).
            IDs of subjects are between 0 and P-1 (inclusive).

        Returns
        -------
        gamma : np.ndarray
            Marginal posterior distribution of hidden states given the data,
            :math:`q(s_t)`. Shape is (batch_size*sequence_length, n_states).
        xi : np.ndarray
            Joint posterior distribution of hidden states at two consecutive
            time points, :math:`q(s_t, s_{t+1})`. Shape is
            (P, batch_size*sequence_length-1, n_states*n_states).
        """
        #subj_ids = subj_ids.astype(np.uint32) # Ensure we don't get negatives here
        P_subjs = self.trans_prob_subj
        Pi_0_subjs = self.state_probs_t0_subj
        log_B = self.get_log_likelihood_hierarchical(x, subj_ids)
        batch_size, sequence_length, n_states = log_B.shape
        log_B = log_B.transpose(2, 0, 1).reshape(n_states, -1).T
        subj_ids = tf.reshape(subj_ids, [-1])
        gamma, xi = self.baum_welch_log_optimized_hierarchical(log_B, Pi_0_subjs, P_subjs, subj_ids)
        return gamma,xi
    
    @numba.jit
    def baum_welch_log_optimized_hierarchical(self, log_B, Pi_0_subjs, P_subjs, subj_ids):
        """Optimized version using xi_sum directly."""        
        log_prob, fwdlattice = _hmmc.forward_log_hierarchical(Pi_0_subjs, P_subjs, log_B,subj_ids)
        bwdlattice = _hmmc.backward_log_hierarchical(Pi_0_subjs, P_subjs, log_B,subj_ids)
        
        log_gamma = fwdlattice + bwdlattice
        self.log_normalize(log_gamma, axis=1)
        gamma = np.exp(log_gamma)
        
        # Get xi_sum DIRECTLY
        log_xi_sum = _hmmc.compute_log_xi_sum_hierarchical(fwdlattice, P_subjs, bwdlattice, log_B,subj_ids)
        xi_sum = np.exp(log_xi_sum)  # Shape: (N_subj, K, K)
        
        N_subj = xi_sum.shape[0]
        
        # Reshape to match expected format: (N_subj, K*K)
        xi_sum_flat = np.swapaxes(xi_sum, 1, 2).reshape((N_subj, -1))
        
        return gamma, xi_sum_flat
    
    def row_normalize(self, mat, eps=1e-300):
        mat = np.asarray(mat)
        s = mat.sum(axis=-1, keepdims=True)
        return mat / np.maximum(s, eps)
    
 
    def update_trans_prob_optimized_hierarchical(self, gamma, xi_sum_flat,subj_ids):
        """Updated to work with pre-summed xi."""
        
        same_next = (subj_ids[:-1] == subj_ids[1:])          # shape (N-1,)
        K = self.config.n_states
        for p in range(self.config.n_subjects):
            if(np.all(xi_sum_flat[p]==0)):
                continue # We can't update this subject: empty transition counts everywhere
            else:
                p_mask = (subj_ids[:-1] == p) & same_next
                # xi_sum_flat is already summed over time, just reshape
                xi_p = xi_sum_flat[p].reshape(K, K).T            # (K,K)
                denom = gamma[:-1][p_mask].sum(axis=0)           # (K,)

                phi_interim = (xi_p + EPS) / (denom[:, None] + K * EPS)  # (K,K), row-wise normalization “in expectation”
                # Ensure exact row stochastic
                phi_interim /= phi_interim.sum(axis=1, keepdims=True)

                
                # Stochastic update of transition matrix
                self.trans_prob[p] = (1 - self.rho) * self.trans_prob[p] + self.rho * phi_interim
                self.trans_prob[p] /= self.trans_prob[p].sum(axis=1, keepdims=True)
    
    def fit(
        self,
        dataset,
        subject_ids,
        epochs=None,
        use_tqdm=False,
        checkpoint_freq=None,
        save_filepath=None,
        dfo_tol=None,
        verbose=1,
        **kwargs,
    ):
        """Fit model to a dataset.

        Iterates between:

        - Baum-Welch updates of latent variable time courses and transition
          probability matrix.
        - TensorFlow updates of observation model parameters.

        Parameters
        ----------
        dataset : tf.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        subject_ids: np.ndarray
            Array of subjects (0 to P-1) of shape ntimepoints (one array per subject)
        epochs : int, optional
            Number of epochs.
        use_tqdm : bool, optional
            Should we use :code:`tqdm` to display a progress bar?
        checkpoint_freq : int, optional
            Frequency (in epochs) of saving model checkpoints.
        save_filepath : str, optional
            Path to save the model.
        dfo_tol : float, optional
            When the maximum fractional occupancy change (from epoch to epoch)
            is less than this value, we stop the training. If :code:`None`
            there is no early stopping.
        verbose : int, optional
            Verbosity level. :code:`0=silent`.
        kwargs : keyword arguments, optional
            Keyword arguments for the TensorFlow observation model training.
            These keywords arguments will be passed to :code:`self.model.fit()`.

        Returns
        -------
        history : dict
            Dictionary with history of the loss, learning rates (:code:`lr`
            and :code:`rho`) and fractional occupancies during training.
        """
        if epochs is None:
            epochs = self.config.n_epochs

        if checkpoint_freq is not None:
            checkpoint = tf.train.Checkpoint(
                model=self.model, optimizer=self.model.optimizer
            )
            if save_filepath is None:
                save_filepath = "tmp"
            self.save_config(save_filepath)
            checkpoint_dir = f"{save_filepath}/checkpoints"
            checkpoint_prefix = f"{checkpoint_dir}/ckpt"

        if dfo_tol is None:
            dfo_tol = 0

        # Make a TensorFlow Dataset
        if isinstance(dataset, Data):
            sfreq = dataset.sampling_frequency
        else:
            sfreq = 1.0
        import subprocess
        import psutil
        import os
        process = psutil.Process(os.getpid())
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])
        gpu_memory_used = gpu_info.decode("utf-8").split("\n")[0]
        _logger.info(f"GPU memory used before make dataset: {gpu_memory_used} MB")
        
        mem_usage_mb = process.memory_info().rss / 1024 / 1024
        _logger.info(f"Memory usage before make dataset: {mem_usage_mb:.2f} MB")
    
        # Here fuse the subject IDs with the data such that last channel = subject IDs
        # This is so that batches will maintain proper ordering between the two arrays seamlessly.
        assert len(dataset.arrays) == len(subject_ids), "Dataset and subject IDs should contain the same number of arrays"
        dataset.arrays = [np.concatenate((dataset.arrays[i],subject_ids[i][:,None]),axis=1) for i in range(len(dataset.arrays))]
        
        #print("Dataset array shape", dataset.arrays[0].shape)
    
        dataset = self.make_dataset(dataset, shuffle=False, concatenate=True)
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])
        gpu_memory_used = gpu_info.decode("utf-8").split("\n")[0]
        _logger.info(f"GPU memory used after make dataset: {gpu_memory_used} MB")
        
        mem_usage_mb = process.memory_info().rss / 1024 / 1024
        _logger.info(f"Memory usage after make dataset: {mem_usage_mb:.2f} MB")
        # Set static loss scaling factor (Sets bash size in model)
        self.set_static_loss_scaling_factor(dataset)
        
        
        # Training curves
        history = {"loss": [], "rho": [], "lr": [], "fo": [], "max_dfo": []}

        from keras.callbacks import EarlyStopping

        # create this once, before the loop
        stopper = EarlyStopping(monitor='loss', min_delta=1e-4, patience=20, verbose=1)
        stopper.set_model(self.model)
        stopper.on_train_begin()
        
        #import time

        # Loop through epochs
        if use_tqdm:
            _range = trange(epochs)
        else:
            _range = range(epochs)
        for n in _range:
            # Setup a progress bar for this epoch
            if verbose > 0 and not use_tqdm:
                print("Epoch {}/{}".format(n + 1, epochs))
                pb_i = utils.Progbar(dtf.get_n_batches(dataset))

            # Update rho
            self._update_rho(n)

            # Set learning rate for the observation model
            lr = self.config.learning_rate * np.exp(
                -self.config.observation_update_decay * n
            )
            backend.set_value(self.model.optimizer.lr, lr)

            # Loop over batches
            loss = []
            occupancies = []
            for element in dataset:
                x_tot = self._unpack_inputs(element)
                subj_ids = tf.cast(x_tot[:,:,-1:], tf.uint32) # Cast to unsigned int 32
                x = x_tot[:,:,:-1]
                
                #print("X.shape", x.shape)
                #print("subj_ids.shape", subj_ids.shape)
                
                # Get the gamma tcs, which must account for subject IDs
                gamma, xi_sum = self.get_posterior(x,subj_ids)

                # Update transition probability matrix
                if self.config.learn_trans_prob:
                    self.update_trans_prob_optimized_hierarchical(gamma, xi_sum, subj_ids)

                # Calculate fractional occupancy
                #stc = modes.argmax_time_courses(gamma)
                #occupancies.append(np.sum(stc, axis=0))

                # Reshape gamma: (batch_size*sequence_length, n_states)
                # -> (batch_size, sequence_length, n_states)
                gamma = gamma.reshape(x.shape[0], x.shape[1], -1)

                # Convert to tensor: avoids reconverting x over and over!
                gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
                # Update observation model
                
                x = tf.cast(x, tf.float32)
                #print(x.dtype)
                #print(gamma.dtype)
                x_and_gamma = tf.concat([x, gamma], axis=2)
                h = None
                #end_other = time.time()
                #start_fit_params = time.time()
                h = self.model.train_on_batch({
                        'data_inputs': x_and_gamma,
                        'subject_id_inputs': subj_ids,  # Shape (B, T)
                    }, return_dict=True, **kwargs)
                #end_fit_params = time.time()
                
                #print(f"Posterior compute time: {end_post - start_post}")
                #print(f"Intermediate updates and conversions compute time: {end_other - start_other}")
                #print(f"Fit params compute time: {end_fit_params - start_fit_params}")

                # Get new loss
                l = float(h["loss"])
                #l = h.history["loss"][0]
                if np.isnan(l):
                    _logger.error("Training failed!")
                    return
                loss.append(l)

                if verbose > 0:
                    # Update progress bar
                    if use_tqdm:
                        _range.set_postfix(rho=self.rho, lr=lr, loss=l)
                    else:
                        pb_i.add(
                            1,
                            values=[("rho", self.rho), ("lr", lr), ("loss", l)],
                        )
            gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])
            gpu_memory_used = gpu_info.decode("utf-8").split("\n")[0]
            _logger.info(f"GPU memory used after single iter: {gpu_memory_used} MB")
            
            mem_usage_mb = process.memory_info().rss / 1024 / 1024
            _logger.info(f"Memory usage after single iter: {mem_usage_mb:.2f} MB")

            history["loss"].append(np.mean(loss))
            history["rho"].append(self.rho)
            history["lr"].append(lr)

            occupancy = np.sum(occupancies, axis=0)
            fo = occupancy / np.sum(occupancy)
            history["fo"].append(fo)

            # Save model checkpoint
            if checkpoint_freq is not None and (n + 1) % checkpoint_freq == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            # How much has the fractional occupancy changed?
            if len(history["fo"]) == 1:
                max_dfo = np.max(
                    np.abs(history["fo"][-1] - np.zeros_like(history["fo"][-1]))
                )
            else:
                max_dfo = np.max(np.abs(history["fo"][-1] - history["fo"][-2]))
            history["max_dfo"].append(max_dfo)
            if dfo_tol > 0:
                print(f"Max change in FO: {max_dfo}")
            if max_dfo < dfo_tol:
                break
            
            logs = {'loss': history['loss'][-1]}
            stopper.on_epoch_end(n,logs)
            if stopper.stopped_epoch > 0:
                print(f"Stopping at epoch {n}")
                break
            gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])
            gpu_memory_used = gpu_info.decode("utf-8").split("\n")[0]
            _logger.info(f"GPU memory used after end of iter: {gpu_memory_used} MB")
            
            mem_usage_mb = process.memory_info().rss / 1024 / 1024
            _logger.info(f"Memory usage after end of iter: {mem_usage_mb:.2f} MB")
        if checkpoint_freq is not None:
            np.save(f"{save_filepath}/trans_prob.npy", self.trans_prob)

        if use_tqdm:
            _range.close()

        return history
