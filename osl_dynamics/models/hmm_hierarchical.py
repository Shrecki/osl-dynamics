from hmm import Config, Model
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
            init_L = tf.linalg.cholesky(tf.convert_to_tensor(config.initial_covariances[m], tf.float32)) if config.initial_covariances[m] is not None else None

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
        
        data, gamma = tf.split(data_inputs, [config.n_channels, config.n_states], axis=2)
        static_loss_scaling_factor = static_loss_scaling_factor_layer(data)
        mu_pop = means_pop_layer(data, static_loss_scaling_factor=static_loss_scaling_factor)
        D_pop = trils_pop_layer(data, static_loss_scaling_factor=static_loss_scaling_factor)
        mu_subj_list = []
        D_subj_list = []
        for m in range(config.n_subjects):
            mu_m = means_subj_layers[m](data, static_loss_scaling_factor=static_loss_scaling_factor)
            D_m = trils_subj_layers[m](data, static_loss_scaling_factor=static_loss_scaling_factor)
            mu_subj_list.append(mu_m)
            D_subj_list.append(D_m)
        mu_subj = tf.stack(mu_subj_list, axis=0)  # (M, K, D)
        D_subj = tf.stack(D_subj_list, axis=0)    # (M, K, D, D)
