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
    dtype = x.dtype
    x = tf.convert_to_tensor(x, dtype)
    mu_pop = tf.convert_to_tensor(mu_pop, dtype)
    L_pop = tf.convert_to_tensor(L_pop, dtype)
    mu_subj = tf.convert_to_tensor(mu_subj, dtype)
    L_subj = tf.convert_to_tensor(L_subj, dtype)
    gamma = tf.convert_to_tensor(gamma, dtype)
    subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
    lambda_mu = tf.cast(lambda_mu, dtype)
    lambda_L = tf.cast(lambda_L, dtype)
    n_subjects = tf.cast(n_subjects, dtype)
    
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    D = tf.shape(x)[2]
    K = tf.shape(mu_pop)[0]
    P = tf.shape(mu_subj)[0]
    
    # Get unique subjects in batch
    unique_subjects, _ = tf.unique(subject_ids)
    n_unique = tf.cast(tf.shape(unique_subjects)[0], dtype)
    
    # Gather subject-specific parameters for this batch
    mu_batch = tf.gather(mu_subj, subject_ids, axis=0)  # (B, K, D)
    L_batch = tf.gather(L_subj, subject_ids, axis=0)    # (B, K, D, D)
    
    # =====================
    # Forward pass: compute NLL
    # =====================
    
    # Compute log-likelihood using subject-specific parameters
    # Expand x for broadcasting: (B, T, 1, D)
    x_expanded = tf.expand_dims(x, axis=2)
    
    # Expand parameters: (B, 1, K, D) and (B, 1, K, D, D)
    mu_expanded = tf.expand_dims(mu_batch, axis=1)
    L_expanded = tf.expand_dims(L_batch, axis=1)
    
    mvn = tfd.MultivariateNormalTriL(
        loc=mu_expanded,
        scale_tril=L_expanded,
        allow_nan_stats=False
    )
    
    log_probs = mvn.log_prob(x_expanded)  # (B, T, K)
    
    # Weighted sum over states
    ll_bt = tf.reduce_sum(gamma * log_probs, axis=-1)  # (B, T)
    nll = -tf.reduce_mean(ll_bt)  # scalar
    
    # =====================
    # Regularization loss (only for subjects in batch, scaled)
    # =====================
    
    def compute_reg_for_subject(subj_id):
        mu_s = tf.gather(mu_subj, subj_id, axis=0)  # (K, D)
        L_s = tf.gather(L_subj, subj_id, axis=0)    # (K, D, D)
        
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
    
    # Scale by (n_unique / n_subjects) to normalize
    scale = n_unique / n_subjects
    reg_loss = scale * (lambda_mu / 2.0 * total_mu_reg + lambda_L / 2.0 * total_L_reg)
    
    total_loss = nll + reg_loss
    
    # =====================
    # Custom gradient
    # =====================
    
    def grad(dy):
        dy = tf.cast(dy, dtype)
        
        # Normalization for mean reduction over (B, T)
        norm = tf.cast(B * T, dtype)
        w = gamma / norm  # (B, T, K)
        
        # Residuals per batch element: r[b,t,k] = x[b,t] - mu_batch[b,k]
        r = x[:, :, None, :] - mu_batch[:, None, :, :]  # (B, T, K, D)
        
        # Compute L^{-1} r for each batch element
        # Reshape for batched triangular solve
        r_flat = tf.reshape(r, [B * T * K, D, 1])
        L_batch_tiled = tf.tile(L_batch[:, None, :, :, :], [1, T, 1, 1, 1])  # (B, T, K, D, D)
        L_flat = tf.reshape(L_batch_tiled, [B * T * K, D, D])
        
        y_flat = tf.linalg.triangular_solve(L_flat, r_flat, lower=True)
        y = tf.reshape(y_flat, [B, T, K, D])  # (B, T, K, D)
        
        # Compute L^{-T} for each batch element
        I_eye = tf.eye(D, dtype=dtype)
        I_batch = tf.tile(I_eye[None, None, :, :], [B, K, 1, 1])  # (B, K, D, D)
        L_batch_T = tf.transpose(L_batch, [0, 1, 3, 2])  # (B, K, D, D)
        LinvT_batch = tf.linalg.triangular_solve(L_batch_T, I_batch, lower=False)  # (B, K, D, D)
        
        # u = Sigma^{-1} r = L^{-T} L^{-1} r = L^{-T} y
        u = tf.einsum("bkij,btkj->btki", LinvT_batch, y)  # (B, T, K, D)
        
        # =====================
        # Gradient for mu_subj: accumulate per subject
        # =====================
        
        # Per-batch gradient contribution: -w * u, shape (B, T, K, D)
        grad_mu_per_sample = -w[:, :, :, None] * u  # (B, T, K, D)
        
        # Sum over time: (B, K, D)
        grad_mu_per_batch = tf.reduce_sum(grad_mu_per_sample, axis=1)
        
        # Scatter-add to subject gradients
        # Initialize gradient tensor for all subjects
        grad_mu_subj_nll = tf.zeros_like(mu_subj)  # (P, K, D)
        
        # Use tensor_scatter_nd_add to accumulate gradients per subject
        indices = tf.expand_dims(subject_ids, 1)  # (B, 1)
        grad_mu_subj_nll = tf.tensor_scatter_nd_add(
            grad_mu_subj_nll, indices, grad_mu_per_batch
        )
        
        # =====================
        # Gradient for L_subj: accumulate per subject
        # =====================
        
        # Sufficient statistics per batch element
        # S[b,k] = sum_t w[b,t,k] * r[b,t,k] * r[b,t,k]^T
        S_batch = tf.einsum("btk,btkd,btkf->bkdf", w, r, r)  # (B, K, D, D)
        
        # N[b,k] = sum_t w[b,t,k]
        Nk_batch = tf.reduce_sum(w, axis=1)  # (B, K)
        
        # Compute gradient: tril(L^{-T} (N*I - L^{-1} S L^{-T}))
        # tmp = L^{-1} S
        tmp = tf.linalg.triangular_solve(L_batch, S_batch, lower=True)  # (B, K, D, D)
        # A = tmp @ L^{-T} = L^{-1} S L^{-T}
        A = tf.matmul(tmp, LinvT_batch)  # (B, K, D, D)
        
        I_K = tf.eye(D, dtype=dtype)[None, None, :, :]  # (1, 1, D, D)
        I_K = tf.tile(I_K, [B, K, 1, 1])  # (B, K, D, D)
        
        # gL = L^{-T} @ (Nk * I - A)
        grad_L_per_batch = tf.matmul(LinvT_batch, Nk_batch[:, :, None, None] * I_K - A)
        grad_L_per_batch = tf.linalg.band_part(grad_L_per_batch, -1, 0)  # Lower triangular
        
        # Scatter-add to subject gradients
        grad_L_subj_nll = tf.zeros_like(L_subj)  # (P, K, D, D)
        grad_L_subj_nll = tf.tensor_scatter_nd_add(
            grad_L_subj_nll, indices, grad_L_per_batch
        )
        
        # =====================
        # Add regularization gradients (only for subjects in batch)
        # =====================
        
        # Create mask for subjects in batch
        subject_mask = tf.scatter_nd(
            tf.expand_dims(unique_subjects, 1),
            tf.ones_like(unique_subjects, dtype=dtype),
            [P]
        )  # (P,) with 1s for present subjects
        
        # Gradient from regularization for mu_subj: 
        # d/d(mu_subj) [lambda_mu/2 * ||mu_subj - mu_pop||^2] = lambda_mu * (mu_subj - mu_pop)
        # But only for subjects in batch, scaled by (n_unique / n_subjects)
        reg_scale = scale  # n_unique / n_subjects
        
        grad_mu_subj_reg = lambda_mu * (mu_subj - mu_pop[None, :, :])  # (P, K, D)
        grad_mu_subj_reg = grad_mu_subj_reg * subject_mask[:, None, None]  # mask out absent subjects
        grad_mu_subj_reg = grad_mu_subj_reg * reg_scale
        
        grad_L_subj_reg = lambda_L * (L_subj - L_pop[None, :, :, :])  # (P, K, D, D)
        grad_L_subj_reg = grad_L_subj_reg * subject_mask[:, None, None, None]
        grad_L_subj_reg = grad_L_subj_reg * reg_scale
        
        # Total subject gradients
        grad_mu_subj = dy * (grad_mu_subj_nll + grad_mu_subj_reg)
        grad_L_subj = dy * (grad_L_subj_nll + grad_L_subj_reg)
        
        # =====================
        # Gradient for population parameters
        # Only from subjects in batch, scaled
        # =====================
        
        # grad w.r.t mu_pop = -lambda_mu * sum_{p in batch} (mu_p - mu_pop) * scale
        #                   = lambda_mu * sum_{p in batch} (mu_pop - mu_p) * scale
        mu_diff_masked = (mu_subj - mu_pop[None, :, :]) * subject_mask[:, None, None]
        grad_mu_pop = -dy * lambda_mu * tf.reduce_sum(mu_diff_masked, axis=0) * reg_scale
        
        # grad w.r.t L_pop = -lambda_L * sum_{p in batch} (L_p - L_pop) * scale
        L_diff_masked = (L_subj - L_pop[None, :, :, :]) * subject_mask[:, None, None, None]
        grad_L_pop = -dy * lambda_L * tf.reduce_sum(L_diff_masked, axis=0) * reg_scale
        
        # Return gradients in same order as inputs
        # (x, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids, lambda_mu, lambda_L, n_subjects)
        return (None, grad_mu_pop, grad_L_pop, grad_mu_subj, grad_L_subj, None, None, None, None, None)
    
    return total_loss, grad


@tf.custom_gradient
def hierarchical_categorical_nll_sum_custom_grad(
    x, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids, lambda_mu, lambda_L, n_subjects
):
    """
    Same as above but with 'sum' reduction over time (mean over batch only).
    """
    dtype = x.dtype
    x = tf.convert_to_tensor(x, dtype)
    mu_pop = tf.convert_to_tensor(mu_pop, dtype)
    L_pop = tf.convert_to_tensor(L_pop, dtype)
    mu_subj = tf.convert_to_tensor(mu_subj, dtype)
    L_subj = tf.convert_to_tensor(L_subj, dtype)
    gamma = tf.convert_to_tensor(gamma, dtype)
    subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
    lambda_mu = tf.cast(lambda_mu, dtype)
    lambda_L = tf.cast(lambda_L, dtype)
    n_subjects = tf.cast(n_subjects, dtype)
    
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    D = tf.shape(x)[2]
    K = tf.shape(mu_pop)[0]
    P = tf.shape(mu_subj)[0]
    
    # Get unique subjects in batch
    unique_subjects, _ = tf.unique(subject_ids)
    n_unique = tf.cast(tf.shape(unique_subjects)[0], dtype)
    
    # Gather subject-specific parameters for this batch
    mu_batch = tf.gather(mu_subj, subject_ids, axis=0)  # (B, K, D)
    L_batch = tf.gather(L_subj, subject_ids, axis=0)    # (B, K, D, D)
    
    # =====================
    # Forward pass: compute NLL
    # =====================
    
    x_expanded = tf.expand_dims(x, axis=2)
    mu_expanded = tf.expand_dims(mu_batch, axis=1)
    L_expanded = tf.expand_dims(L_batch, axis=1)
    
    mvn = tfd.MultivariateNormalTriL(
        loc=mu_expanded,
        scale_tril=L_expanded,
        allow_nan_stats=False
    )
    
    log_probs = mvn.log_prob(x_expanded)  # (B, T, K)
    ll_bt = tf.reduce_sum(gamma * log_probs, axis=-1)  # (B, T)
    
    # Sum over time, mean over batch
    ll_scalar = tf.reduce_mean(tf.reduce_sum(ll_bt, axis=1))
    nll = -ll_scalar
    
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
    
    scale = n_unique / n_subjects
    reg_loss = scale * (lambda_mu / 2.0 * total_mu_reg + lambda_L / 2.0 * total_L_reg)
    
    total_loss = nll + reg_loss
    
    # =====================
    # Custom gradient
    # =====================
    
    def grad(dy):
        dy = tf.cast(dy, dtype)
        
        # Normalization: sum over time, mean over batch -> divide by B only
        norm = tf.cast(B, dtype)
        w = gamma / norm  # (B, T, K)
        
        r = x[:, :, None, :] - mu_batch[:, None, :, :]  # (B, T, K, D)
        
        r_flat = tf.reshape(r, [B * T * K, D, 1])
        L_batch_tiled = tf.tile(L_batch[:, None, :, :, :], [1, T, 1, 1, 1])
        L_flat = tf.reshape(L_batch_tiled, [B * T * K, D, D])
        
        y_flat = tf.linalg.triangular_solve(L_flat, r_flat, lower=True)
        y = tf.reshape(y_flat, [B, T, K, D])
        
        I_eye = tf.eye(D, dtype=dtype)
        I_batch = tf.tile(I_eye[None, None, :, :], [B, K, 1, 1])
        L_batch_T = tf.transpose(L_batch, [0, 1, 3, 2])
        LinvT_batch = tf.linalg.triangular_solve(L_batch_T, I_batch, lower=False)
        
        u = tf.einsum("bkij,btkj->btki", LinvT_batch, y)
        
        # Gradient for mu_subj
        grad_mu_per_sample = -w[:, :, :, None] * u
        grad_mu_per_batch = tf.reduce_sum(grad_mu_per_sample, axis=1)
        
        grad_mu_subj_nll = tf.zeros_like(mu_subj)
        indices = tf.expand_dims(subject_ids, 1)
        grad_mu_subj_nll = tf.tensor_scatter_nd_add(
            grad_mu_subj_nll, indices, grad_mu_per_batch
        )
        
        # Gradient for L_subj
        S_batch = tf.einsum("btk,btkd,btkf->bkdf", w, r, r)
        Nk_batch = tf.reduce_sum(w, axis=1)
        
        tmp = tf.linalg.triangular_solve(L_batch, S_batch, lower=True)
        A = tf.matmul(tmp, LinvT_batch)
        
        I_K = tf.eye(D, dtype=dtype)[None, None, :, :]
        I_K = tf.tile(I_K, [B, K, 1, 1])
        
        grad_L_per_batch = tf.matmul(LinvT_batch, Nk_batch[:, :, None, None] * I_K - A)
        grad_L_per_batch = tf.linalg.band_part(grad_L_per_batch, -1, 0)
        
        grad_L_subj_nll = tf.zeros_like(L_subj)
        grad_L_subj_nll = tf.tensor_scatter_nd_add(
            grad_L_subj_nll, indices, grad_L_per_batch
        )
        
        # Regularization gradients
        subject_mask = tf.scatter_nd(
            tf.expand_dims(unique_subjects, 1),
            tf.ones_like(unique_subjects, dtype=dtype),
            [P]
        )
        
        reg_scale = scale
        
        grad_mu_subj_reg = lambda_mu * (mu_subj - mu_pop[None, :, :])
        grad_mu_subj_reg = grad_mu_subj_reg * subject_mask[:, None, None] * reg_scale
        
        grad_L_subj_reg = lambda_L * (L_subj - L_pop[None, :, :, :])
        grad_L_subj_reg = grad_L_subj_reg * subject_mask[:, None, None, None] * reg_scale
        
        grad_mu_subj = dy * (grad_mu_subj_nll + grad_mu_subj_reg)
        grad_L_subj = dy * (grad_L_subj_nll + grad_L_subj_reg)
        
        # Population gradients
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
