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
import tensorflow as tf
@dataclass
class HierarchicalConfig(Config):
    n_subjects: int = None
    lambda_mu: float = 1.0
    lambda_L: float = 1.0
    
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
    subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
    lambda_mu = tf.cast(lambda_mu, dtype)
    lambda_L = tf.cast(lambda_L, dtype)

    n_subjects = tf.cast(n_subjects, dtype)
    
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    D = tf.shape(x)[2]
    K = tf.shape(mu_pop)[0]
    P = tf.shape(mu_subj)[0]
    
    
    # =====================
    # Sort subjects by their IDs to break into consecutive segments the data (for emission forward, no need to have consecutive samples)
    # =====================
    unique_subjects, row_splits, x_sorted, gamma_sorted, _ = sort_and_rowsplit_by_subject(
        x, gamma, subject_ids
    )

    N = tf.cast(B*T, dtype)
    S = tf.shape(unique_subjects)[0]
    
    # =====================
    # Forward pass: compute NLL
    # =====================
    nll_sum = tf.zeros([], dtype=dtype)  # accumulate total negative log-likelihood numerator
    
    def per_subject_ll(i):
        start = row_splits[i]
        stop  = row_splits[i + 1]
        sid   = unique_subjects[i]

        x_block = x_sorted[start:stop]         # (N_i, D)
        g_block = gamma_sorted[start:stop]     # (N_i, K)

        mu_i = tf.gather(mu_subj, sid, axis=0) # (K, D)
        L_i  = tf.gather(L_subj,  sid, axis=0) # (K, D, D)

        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mu_i,
            scale_tril=L_i,
            allow_nan_stats=False
        )
        logp = mvn.log_prob(x_block[:, None, :])          # (N_i, K)

        # return NEGATIVE contribution (so we can sum directly)
        return -tf.reduce_sum(g_block * logp)             # scalar

    nll_sum = tf.reduce_sum(
        tf.map_fn(per_subject_ll, tf.range(S), fn_output_signature=dtype)
    )
    
    nll = nll_sum/ N
    # =====================
    # Regularization loss
    # =====================
    
    def compute_reg_for_subject(subj_id):
        mu_s = tf.gather(mu_subj, subj_id, axis=0)
        L_s = tf.gather(L_subj, subj_id, axis=0)
        mu_diff = mu_s - mu_pop
        L_diff = L_s - L_pop
        mu_reg = tf.reduce_sum(tf.square(mu_diff))
        L_reg = tf.reduce_sum(tf.square(L_diff))
        return mu_reg, L_reg
    
    mu_regs, L_regs = tf.map_fn(
        compute_reg_for_subject,
        unique_subjects,
        fn_output_signature=(
            tf.TensorSpec(shape=(), dtype=dtype),
            tf.TensorSpec(shape=(), dtype=dtype)
        )
    )
    
    total_mu_reg = tf.reduce_sum(mu_regs)
    total_L_reg = tf.reduce_sum(L_regs)
    
    scale = tf.cast(S, dtype) / n_subjects
    reg_loss = scale * (lambda_mu / 2.0 * total_mu_reg + lambda_L / 2.0 * total_L_reg)
    
    total_loss = nll + reg_loss
    
    # =====================
    # Custom gradient
    # =====================
    
    def grad(dy):
        dy = tf.cast(dy, dtype)

        # Mean over all timepoints => norm = N = B*T
        # If you want "sum over time, mean over batch", set norm = tf.cast(B, dtype)
        norm = N

        grad_mu_subj_nll = tf.zeros_like(mu_subj)  # (P,K,D)
        grad_L_subj_nll  = tf.zeros_like(L_subj)   # (P,K,D,D)

        I = tf.eye(D, dtype=dtype)[None, :, :]     # (1,D,D), will broadcast

        # Loop over unique subjects (handful => OK)
        # If you later trace this, replace with tf.map_fn over tf.range(S).
        for i in range(int(unique_subjects.shape[0])):  # eager-friendly
            start = row_splits[i]
            stop  = row_splits[i + 1]
            sid   = unique_subjects[i]  # scalar int32

            x_block = x_sorted[start:stop]         # (N_i, D)
            g_block = gamma_sorted[start:stop]     # (N_i, K)

            mu_i = tf.gather(mu_subj, sid, axis=0) # (K, D)
            L_i  = tf.gather(L_subj,  sid, axis=0) # (K, D, D)

            # weights
            w = g_block / norm                     # (N_i, K)

            # residuals: (N_i, K, D)
            r = x_block[:, None, :] - mu_i[None, :, :]

            # y = L^{-1} r  -> (N_i, K, D)
            y = tf.linalg.triangular_solve(L_i[None, ...], r[..., None], lower=True)
            y = tf.squeeze(y, axis=-1)

            # u = L^{-T} y  -> (N_i, K, D)
            u = tf.linalg.triangular_solve(
                tf.linalg.matrix_transpose(L_i)[None, ...],
                y[..., None],
                lower=False
            )
            u = tf.squeeze(u, axis=-1)

            # ---- mu grad for this subject (K,D)
            gmu = -tf.einsum("nk,nkd->kd", w, u)  # sum over N_i

            # ---- L grad for this subject (K,D,D)
            # S_k = sum_n w[n,k] r[n,k] r[n,k]^T
            S_k = tf.einsum("nk,nkd,nkf->kdf", w, r, r)   # (K,D,D)
            Nk  = tf.reduce_sum(w, axis=0)                # (K,)

            # LinvT = L^{-T} (K,D,D)
            LinvT = tf.linalg.triangular_solve(
                tf.linalg.matrix_transpose(L_i),
                tf.eye(D, dtype=dtype)[None, :, :],
                lower=False
            )  # (K,D,D)

            tmp = tf.linalg.triangular_solve(L_i, S_k, lower=True)  # (K,D,D) = L^{-1} S
            A   = tf.matmul(tmp, LinvT)                             # (K,D,D) = L^{-1} S L^{-T}

            NI_minus_A = Nk[:, None, None] * I - A                  # (K,D,D)
            gL = tf.matmul(LinvT, NI_minus_A)                       # (K,D,D)
            gL = tf.linalg.band_part(gL, -1, 0)                     # keep lower tri

            # scatter-add into full subject tensors
            idx = tf.reshape(sid, [1, 1])  # shape (1,1)
            grad_mu_subj_nll = tf.tensor_scatter_nd_add(grad_mu_subj_nll, idx, gmu[None, ...])
            grad_L_subj_nll  = tf.tensor_scatter_nd_add(grad_L_subj_nll,  idx, gL[None, ...])

        # ---------------------
        # Regularization grads (same as before)
        # ---------------------
        subject_mask = tf.scatter_nd(
            tf.expand_dims(unique_subjects, 1),
            tf.ones_like(unique_subjects, dtype=dtype),
            [P]
        )  # (P,)

        reg_scale = scale

        grad_mu_subj_reg = lambda_mu * (mu_subj - mu_pop[None, :, :])
        grad_mu_subj_reg *= subject_mask[:, None, None] * reg_scale

        grad_L_subj_reg = lambda_L * (L_subj - L_pop[None, :, :, :])
        grad_L_subj_reg *= subject_mask[:, None, None, None] * reg_scale

        grad_mu_subj = dy * (grad_mu_subj_nll + grad_mu_subj_reg)
        grad_L_subj  = dy * (grad_L_subj_nll  + grad_L_subj_reg)

        # Population grads (from reg only, same as before)
        mu_diff_masked = (mu_subj - mu_pop[None, :, :]) * subject_mask[:, None, None]
        grad_mu_pop = -dy * lambda_mu * tf.reduce_sum(mu_diff_masked, axis=0) * reg_scale

        L_diff_masked = (L_subj - L_pop[None, :, :, :]) * subject_mask[:, None, None, None]
        grad_L_pop = -dy * lambda_L * tf.reduce_sum(L_diff_masked, axis=0) * reg_scale

        return (None, grad_mu_pop, grad_L_pop, grad_mu_subj, grad_L_subj, None, None, None, None, None)
    return total_loss, grad


@tf.custom_gradient
def hierarchical_categorical_nll_sum_custom_grad(
    x, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids, lambda_mu, lambda_L, n_subjects
):
    """
    Same as above but with 'sum' reduction over time (mean over batch only).
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
    
    # =====================
    # Sort subjects by their IDs to break into consecutive segments the data (for emission forward, no need to have consecutive samples)
    # =====================
    unique_subjects, row_splits, x_sorted, gamma_sorted, _ = sort_and_rowsplit_by_subject(
        x, gamma, subject_ids
    )

    N = tf.cast(B, dtype)
    #N = tf.cast(tf.shape(x_sorted)[0], dtype)
    S = tf.shape(unique_subjects)[0]
    
    # =====================
    # Forward pass: compute NLL
    # =====================
    nll_sum = tf.zeros([], dtype=dtype)  # accumulate total negative log-likelihood numerator
    def per_subject_ll(i):
        start = row_splits[i]
        stop  = row_splits[i + 1]
        sid   = unique_subjects[i]

        x_block = x_sorted[start:stop]         # (N_i, D)
        g_block = gamma_sorted[start:stop]     # (N_i, K)

        mu_i = tf.gather(mu_subj, sid, axis=0) # (K, D)
        L_i  = tf.gather(L_subj,  sid, axis=0) # (K, D, D)

        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mu_i,
            scale_tril=L_i,
            allow_nan_stats=False
        )
        logp = mvn.log_prob(x_block[:, None, :])          # (N_i, K)

        # return NEGATIVE contribution (so we can sum directly)
        return -tf.reduce_sum(g_block * logp)             # scalar

    nll_sum = tf.reduce_sum(
        tf.map_fn(per_subject_ll, tf.range(S), fn_output_signature=dtype)
    )
    
    nll = nll_sum/ N
    # =====================
    # Regularization loss
    # =====================
    
    def compute_reg_for_subject(subj_id):
        mu_s = tf.gather(mu_subj, subj_id, axis=0)
        L_s = tf.gather(L_subj, subj_id, axis=0)
        mu_diff = mu_s - mu_pop
        L_diff = L_s - L_pop
        mu_reg = tf.reduce_sum(tf.square(mu_diff))
        L_reg = tf.reduce_sum(tf.square(L_diff))
        return mu_reg, L_reg
    
    mu_regs, L_regs = tf.map_fn(
        compute_reg_for_subject,
        unique_subjects,
        fn_output_signature=(
            tf.TensorSpec(shape=(), dtype=dtype),
            tf.TensorSpec(shape=(), dtype=dtype)
        )
    )
    
    total_mu_reg = tf.reduce_sum(mu_regs)
    total_L_reg = tf.reduce_sum(L_regs)
    
    scale = tf.cast(S, dtype) / n_subjects
    reg_loss = scale * (lambda_mu / 2.0 * total_mu_reg + lambda_L / 2.0 * total_L_reg)
    
    total_loss = nll + reg_loss
    
    # =====================
    # Custom gradient
    # =====================
    
    def grad(dy):
        dy = tf.cast(dy, dtype)

        # Mean over all timepoints => norm = N = B*T
        # If you want "sum over time, mean over batch", set norm = tf.cast(B, dtype)
        norm = N

        grad_mu_subj_nll = tf.zeros_like(mu_subj)  # (P,K,D)
        grad_L_subj_nll  = tf.zeros_like(L_subj)   # (P,K,D,D)

        I = tf.eye(D, dtype=dtype)[None, :, :]     # (1,D,D), will broadcast

        # Loop over unique subjects (handful => OK)
        # If you later trace this, replace with tf.map_fn over tf.range(S).
        for i in range(int(unique_subjects.shape[0])):  # eager-friendly
            start = row_splits[i]
            stop  = row_splits[i + 1]
            sid   = unique_subjects[i]  # scalar int32

            x_block = x_sorted[start:stop]         # (N_i, D)
            g_block = gamma_sorted[start:stop]     # (N_i, K)

            mu_i = tf.gather(mu_subj, sid, axis=0) # (K, D)
            L_i  = tf.gather(L_subj,  sid, axis=0) # (K, D, D)

            # weights
            w = g_block / norm                     # (N_i, K)

            # residuals: (N_i, K, D)
            r = x_block[:, None, :] - mu_i[None, :, :]

            # y = L^{-1} r  -> (N_i, K, D)
            y = tf.linalg.triangular_solve(L_i[None, ...], r[..., None], lower=True)
            y = tf.squeeze(y, axis=-1)

            # u = L^{-T} y  -> (N_i, K, D)
            u = tf.linalg.triangular_solve(
                tf.linalg.matrix_transpose(L_i)[None, ...],
                y[..., None],
                lower=False
            )
            u = tf.squeeze(u, axis=-1)

            # ---- mu grad for this subject (K,D)
            gmu = -tf.einsum("nk,nkd->kd", w, u)  # sum over N_i

            # ---- L grad for this subject (K,D,D)
            # S_k = sum_n w[n,k] r[n,k] r[n,k]^T
            S_k = tf.einsum("nk,nkd,nkf->kdf", w, r, r)   # (K,D,D)
            Nk  = tf.reduce_sum(w, axis=0)                # (K,)

            # LinvT = L^{-T} (K,D,D)
            LinvT = tf.linalg.triangular_solve(
                tf.linalg.matrix_transpose(L_i),
                tf.eye(D, dtype=dtype)[None, :, :],
                lower=False
            )  # (K,D,D)

            tmp = tf.linalg.triangular_solve(L_i, S_k, lower=True)  # (K,D,D) = L^{-1} S
            A   = tf.matmul(tmp, LinvT)                             # (K,D,D) = L^{-1} S L^{-T}

            NI_minus_A = Nk[:, None, None] * I - A                  # (K,D,D)
            gL = tf.matmul(LinvT, NI_minus_A)                       # (K,D,D)
            gL = tf.linalg.band_part(gL, -1, 0)                     # keep lower tri

            # scatter-add into full subject tensors
            idx = tf.reshape(sid, [1, 1])  # shape (1,1)
            grad_mu_subj_nll = tf.tensor_scatter_nd_add(grad_mu_subj_nll, idx, gmu[None, ...])
            grad_L_subj_nll  = tf.tensor_scatter_nd_add(grad_L_subj_nll,  idx, gL[None, ...])

        # ---------------------
        # Regularization grads (same as before)
        # ---------------------
        subject_mask = tf.scatter_nd(
            tf.expand_dims(unique_subjects, 1),
            tf.ones_like(unique_subjects, dtype=dtype),
            [P]
        )  # (P,)

        reg_scale = scale

        grad_mu_subj_reg = lambda_mu * (mu_subj - mu_pop[None, :, :])
        grad_mu_subj_reg *= subject_mask[:, None, None] * reg_scale

        grad_L_subj_reg = lambda_L * (L_subj - L_pop[None, :, :, :])
        grad_L_subj_reg *= subject_mask[:, None, None, None] * reg_scale

        grad_mu_subj = dy * (grad_mu_subj_nll + grad_mu_subj_reg)
        grad_L_subj  = dy * (grad_L_subj_nll  + grad_L_subj_reg)

        # Population grads (from reg only, same as before)
        mu_diff_masked = (mu_subj - mu_pop[None, :, :]) * subject_mask[:, None, None]
        grad_mu_pop = -dy * lambda_mu * tf.reduce_sum(mu_diff_masked, axis=0) * reg_scale

        L_diff_masked = (L_subj - L_pop[None, :, :, :]) * subject_mask[:, None, None, None]
        grad_L_pop = -dy * lambda_L * tf.reduce_sum(L_diff_masked, axis=0) * reg_scale

        return (None, grad_mu_pop, grad_L_pop, grad_mu_subj, grad_L_subj, None, None, None, None, None)

    
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
    
    def __init__(self, config):
        super().__init__(config)
        
        # Population transition probability
        self.trans_prob_pop = None
        
        # Subject-specific transition probabilities (M matrices)
        self.trans_prob_subj = [None] * config.n_subjects
        
    def set_trans_prob(self, trans_prob, subject_id=None):
        """
        Set transition probability matrix.
        
        Parameters
        ----------
        trans_prob : np.ndarray
            Transition probability matrix
        subject_id : int, optional
            If None, set population matrix. Otherwise set subject-specific matrix.
        """
        if subject_id is None:
            self.trans_prob_pop = trans_prob
        else:
            self.trans_prob_subj[subject_id] = trans_prob

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
            config.initial_means,
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
                    config.initial_means,
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
            init_L = tf.linalg.cholesky(tf.convert_to_tensor(config.initial_covariances[m], tf.float32)) if config.initial_covariances is not None and config.initial_covariances[m] is not None else None
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
