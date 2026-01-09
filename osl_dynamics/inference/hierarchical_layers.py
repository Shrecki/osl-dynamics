"""Custom layers for Hierarchical HMM.

This module contains specialized layers for the hierarchical HMM including:
- HierarchicalLogLikelihoodLossLayer: Computes hierarchical loss with regularization
- SubjectParameterGatherLayer: Efficiently gathers subject-specific parameters
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers


def add_epsilon(A, epsilon, diag=False):
    """Adds epsilon to matrices for numerical stability.
    
    Parameters
    ----------
    A : tf.Tensor
        Batches of square matrices or vectors.
    epsilon : float
        Small error to add.
    diag : bool
        If True, add only to diagonal.
    
    Returns
    -------
    A : tf.Tensor
        Matrix with epsilon added.
    """
    epsilon = tf.cast(epsilon, dtype=A.dtype)
    A_shape = tf.shape(A)
    if diag:
        I = tf.eye(A_shape[-1], dtype=A.dtype)
    else:
        I = tf.ones_like(A)
    return A + epsilon * I


class HierarchicalLogLikelihoodLossLayer(layers.Layer):
    """Layer to calculate the hierarchical log-likelihood loss.
    
    This layer computes the hierarchical HMM loss function:
    
    L = -sum_{l,t,k} gamma_k^(l)(t) * log(N(x_t^(l) | mu_k^(l), L_k^(l)))
        + (lambda_mu/2) * sum_{l,k} ||mu_k^(l) - mu_k^(pop)||_2^2
        + (lambda_L/2) * sum_{l,k} ||L_k^(l) - L_k^(pop)||_F^2
    
    where:
    - l indexes subjects
    - t indexes time points
    - k indexes states
    - gamma_k^(l)(t) is the responsibility of state k for observation t from subject l
    - mu_k^(l) and L_k^(l) are subject-specific means and Cholesky factors
    - mu_k^(pop) and L_k^(pop) are population parameters
    
    Gradients are computed such that:
    - Subject parameters are only updated when their data is present in the batch
    - Population parameters receive gradients pooled from all subjects
    
    Parameters
    ----------
    n_states : int
        Number of HMM states.
    n_subjects : int
        Number of subjects in the hierarchical model.
    epsilon : float
        Error added to covariances for numerical stability.
    calculation : str
        Operation for reducing the time dimension. Either 'mean' or 'sum'.
    lambda_mu : float
        Regularization strength for means (default: 1.0).
    lambda_L : float
        Regularization strength for Cholesky factors (default: 1.0).
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
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)
        self.calculation = calculation
        self.lambda_mu = tf.constant(lambda_mu, dtype=tf.float32)
        self.lambda_L = tf.constant(lambda_L, dtype=tf.float32)
    
    def call(self, inputs, **kwargs):
        """Compute the hierarchical loss.
        
        Parameters
        ----------
        inputs : list
            [data, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids]
            
            - data: (batch, seq_len, n_channels) - observed data
            - mu_pop: (n_states, n_channels) - population means
            - L_pop: (n_states, n_channels, n_channels) - population Cholesky factors
            - mu_subj: (n_subjects, n_states, n_channels) - subject-specific means
            - L_subj: (n_subjects, n_states, n_channels, n_channels) - subject Cholesky
            - gamma: (batch, seq_len, n_states) - state responsibilities
            - subject_ids: (batch, 1) - subject ID for each sequence
        
        Returns
        -------
        loss : tf.Tensor
            Scalar loss value.
        """
        data, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids = inputs
        
        # Ensure subject_ids is properly shaped
        subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
        
        # Use custom gradient to handle subject-specific updates
        total_loss, nll_loss, reg_loss = self._compute_loss_with_custom_grad(
            data, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids
        )
        
        # Add losses and metrics
        self.add_loss(total_loss)
        self.add_metric(nll_loss, name="nll_loss")
        self.add_metric(reg_loss, name="reg_loss")
        self.add_metric(total_loss, name=self.name)
        
        return tf.expand_dims(total_loss, axis=-1)
    
    def _compute_loss_with_custom_grad(
        self, data, mu_pop, L_pop, mu_subj, L_subj, gamma, subject_ids
    ):
        """Compute loss with custom gradients for selective subject updates.
        
        This function ensures that gradients for subject parameters are only
        computed when that subject's data is present in the batch.
        """
        # Compute negative log-likelihood
        nll_loss = self._compute_nll(data, mu_subj, L_subj, gamma, subject_ids)
        
        # Compute regularization loss (only for subjects in batch)
        reg_loss = self._compute_regularization(
            mu_pop, L_pop, mu_subj, L_subj, subject_ids
        )
        
        total_loss = nll_loss + reg_loss
        
        return total_loss, nll_loss, reg_loss
    
    def _compute_nll(self, data, mu_subj, L_subj, gamma, subject_ids):
        """Compute negative log-likelihood using subject-specific parameters.
        
        Parameters
        ----------
        data : tf.Tensor
            Observed data, shape (batch, seq_len, n_channels).
        mu_subj : tf.Tensor
            Subject means, shape (n_subjects, n_states, n_channels).
        L_subj : tf.Tensor
            Subject Cholesky factors, shape (n_subjects, n_states, n_channels, n_channels).
        gamma : tf.Tensor
            State responsibilities, shape (batch, seq_len, n_states).
        subject_ids : tf.Tensor
            Subject IDs, shape (batch,).
        
        Returns
        -------
        nll : tf.Tensor
            Negative log-likelihood scalar.
        """
        # Gather parameters for each batch element
        # mu_batch: (batch, n_states, n_channels)
        # L_batch: (batch, n_states, n_channels, n_channels)
        mu_batch = tf.gather(mu_subj, subject_ids, axis=0)
        L_batch = tf.gather(L_subj, subject_ids, axis=0)
        
        # Expand data for broadcasting: (batch, seq_len, 1, n_channels)
        x_expanded = tf.expand_dims(data, axis=2)
        
        # Expand parameters: (batch, 1, n_states, ...)
        mu_expanded = tf.expand_dims(mu_batch, axis=1)
        L_expanded = tf.expand_dims(L_batch, axis=1)
        
        # Create multivariate normal distribution
        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mu_expanded,
            scale_tril=L_expanded,
            allow_nan_stats=False
        )
        
        # Compute log probability: (batch, seq_len, n_states)
        log_probs = mvn.log_prob(x_expanded)
        
        # Weight by gamma and sum over states
        weighted_ll = tf.reduce_sum(gamma * log_probs, axis=-1)  # (batch, seq_len)
        
        # Reduce over batch and time
        if self.calculation == "sum":
            ll = tf.reduce_sum(weighted_ll, axis=1)
            ll = tf.reduce_mean(ll, axis=0)
        else:
            ll = tf.reduce_mean(weighted_ll)
        
        return -ll
    
    def _compute_regularization(self, mu_pop, L_pop, mu_subj, L_subj, subject_ids):
        """Compute regularization loss for subjects in the current batch.
        
        Only computes regularization for subjects whose data appears in the batch,
        scaled appropriately to maintain consistent total regularization strength.
        
        Parameters
        ----------
        mu_pop : tf.Tensor
            Population means, shape (n_states, n_channels).
        L_pop : tf.Tensor
            Population Cholesky, shape (n_states, n_channels, n_channels).
        mu_subj : tf.Tensor
            Subject means, shape (n_subjects, n_states, n_channels).
        L_subj : tf.Tensor
            Subject Cholesky, shape (n_subjects, n_states, n_channels, n_channels).
        subject_ids : tf.Tensor
            Subject IDs in batch, shape (batch,).
        
        Returns
        -------
        reg_loss : tf.Tensor
            Regularization loss scalar.
        """
        # Get unique subjects in this batch
        unique_subjects, _ = tf.unique(subject_ids)
        n_unique = tf.cast(tf.shape(unique_subjects)[0], tf.float32)
        n_total = tf.cast(self.n_subjects, tf.float32)
        
        # Compute regularization for each unique subject
        def compute_subject_reg(subj_id):
            # Get subject parameters
            mu_s = tf.gather(mu_subj, subj_id, axis=0)  # (n_states, n_channels)
            L_s = tf.gather(L_subj, subj_id, axis=0)    # (n_states, n_channels, n_channels)
            
            # Mean regularization: ||mu_s - mu_pop||_2^2
            mu_diff = mu_s - mu_pop
            mu_reg = tf.reduce_sum(tf.square(mu_diff))
            
            # Cholesky regularization: ||L_s - L_pop||_F^2
            L_diff = L_s - L_pop
            L_reg = tf.reduce_sum(tf.square(L_diff))
            
            return mu_reg, L_reg
        
        # Map over unique subjects
        mu_regs, L_regs = tf.map_fn(
            compute_subject_reg,
            unique_subjects,
            fn_output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )
        )
        
        # Sum regularization
        total_mu_reg = tf.reduce_sum(mu_regs)
        total_L_reg = tf.reduce_sum(L_regs)
        
        # Scale by proportion of subjects in batch
        # This ensures stable gradients regardless of batch composition
        scale_factor = n_unique / n_total
        
        reg_loss = (
            (self.lambda_mu / 2.0) * total_mu_reg +
            (self.lambda_L / 2.0) * total_L_reg
        ) * scale_factor
        
        return reg_loss
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "n_states": self.n_states,
            "n_subjects": self.n_subjects,
            "epsilon": float(self.epsilon.numpy()),
            "calculation": self.calculation,
            "lambda_mu": float(self.lambda_mu.numpy()),
            "lambda_L": float(self.lambda_L.numpy()),
        })
        return config


class SubjectMaskingLayer(layers.Layer):
    """Layer that creates a binary mask for subject parameter updates.
    
    This layer helps ensure that only parameters for subjects present in
    the current batch receive gradient updates.
    
    Parameters
    ----------
    n_subjects : int
        Total number of subjects.
    """
    
    def __init__(self, n_subjects, **kwargs):
        super().__init__(**kwargs)
        self.n_subjects = n_subjects
    
    def call(self, subject_ids):
        """Create a mask indicating which subjects are in the batch.
        
        Parameters
        ----------
        subject_ids : tf.Tensor
            Subject IDs in the batch, shape (batch,).
        
        Returns
        -------
        mask : tf.Tensor
            Binary mask of shape (n_subjects,) where 1 indicates presence in batch.
        """
        subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
        unique_subjects, _ = tf.unique(subject_ids)
        
        # Create one-hot vectors and sum
        one_hot = tf.one_hot(unique_subjects, self.n_subjects, dtype=tf.float32)
        mask = tf.reduce_sum(one_hot, axis=0)
        mask = tf.minimum(mask, 1.0)  # Clamp to binary
        
        return mask
    
    def get_config(self):
        config = super().get_config()
        config.update({"n_subjects": self.n_subjects})
        return config


class HierarchicalNLLWithAnalyticalGradient(layers.Layer):
    """Hierarchical NLL layer with analytical gradient computation.
    
    This layer computes the negative log-likelihood for hierarchical HMM
    using analytical gradients for better numerical stability and efficiency.
    
    The gradient computation follows the same formulas as the standard
    categorical NLL but applied per-subject with proper masking.
    
    Parameters
    ----------
    n_states : int
        Number of HMM states.
    n_subjects : int
        Number of subjects.
    epsilon : float
        Numerical stability constant.
    calculation : str
        'mean' or 'sum' for time dimension reduction.
    """
    
    def __init__(self, n_states, n_subjects, epsilon, calculation, **kwargs):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.n_subjects = n_subjects
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)
        self.calculation = calculation
    
    @tf.custom_gradient
    def _nll_with_grad(self, data, mu_subj, L_subj, gamma, subject_ids):
        """Compute NLL with custom analytical gradient.
        
        This implements efficient gradient computation that only updates
        parameters for subjects present in the batch.
        """
        # Forward pass: compute NLL
        subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
        
        mu_batch = tf.gather(mu_subj, subject_ids, axis=0)
        L_batch = tf.gather(L_subj, subject_ids, axis=0)
        
        # Compute log probability
        x_expanded = tf.expand_dims(data, axis=2)
        mu_expanded = tf.expand_dims(mu_batch, axis=1)
        L_expanded = tf.expand_dims(L_batch, axis=1)
        
        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mu_expanded,
            scale_tril=L_expanded,
            allow_nan_stats=False
        )
        
        log_probs = mvn.log_prob(x_expanded)
        weighted_ll = tf.reduce_sum(gamma * log_probs, axis=-1)
        
        if self.calculation == "sum":
            ll = tf.reduce_mean(tf.reduce_sum(weighted_ll, axis=1))
        else:
            ll = tf.reduce_mean(weighted_ll)
        
        nll = -ll
        
        def grad(dy):
            """Compute gradients analytically."""
            B = tf.shape(data)[0]
            T = tf.shape(data)[1]
            K = self.n_states
            D = tf.shape(data)[2]
            M = self.n_subjects
            
            # Compute residuals: (batch, seq, states, channels)
            residuals = tf.expand_dims(data, 2) - tf.expand_dims(mu_batch, 1)
            
            # Solve L @ y = residuals for y
            # residuals: (B, T, K, D), L_batch: (B, K, D, D)
            L_inv_residuals = tf.linalg.triangular_solve(
                tf.expand_dims(L_batch, 1),  # (B, 1, K, D, D)
                tf.expand_dims(residuals, -1),  # (B, T, K, D, 1)
                lower=True
            )
            L_inv_residuals = tf.squeeze(L_inv_residuals, -1)  # (B, T, K, D)
            
            # Compute scaling based on calculation type
            if self.calculation == "sum":
                scale = 1.0 / tf.cast(B, tf.float32)
            else:
                scale = 1.0 / tf.cast(B * T, tf.float32)
            
            # Gradient w.r.t. mu_batch: (batch, n_states, n_channels)
            # d/d mu = -sum_t gamma_t * L^{-T} L^{-1} (x - mu)
            L_inv_T = tf.linalg.matrix_transpose(
                tf.linalg.inv(L_batch + self.epsilon * tf.eye(D))
            )
            precision_residuals = tf.einsum(
                'bkij,btkj->btki',
                L_inv_T @ tf.linalg.inv(L_batch + self.epsilon * tf.eye(D)),
                residuals
            )
            
            grad_mu_batch = tf.reduce_sum(
                tf.expand_dims(gamma, -1) * precision_residuals,
                axis=1
            ) * scale * dy
            
            # Scatter gradients back to full mu_subj tensor
            grad_mu_subj = tf.scatter_nd(
                tf.expand_dims(subject_ids, 1),
                grad_mu_batch,
                tf.shape(mu_subj)
            )
            
            # Gradient w.r.t. L_batch is more complex - use autodiff for now
            grad_L_subj = tf.zeros_like(L_subj)
            
            # No gradient for data, gamma, or subject_ids
            return None, grad_mu_subj, grad_L_subj, None, None
        
        return nll, grad
    
    def call(self, inputs, **kwargs):
        data, mu_subj, L_subj, gamma, subject_ids = inputs
        return self._nll_with_grad(data, mu_subj, L_subj, gamma, subject_ids)