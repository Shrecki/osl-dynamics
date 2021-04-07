"""Model class for a generative model with Gaussian observations.

"""

import logging
from operator import lt
from typing import Tuple, Union

import numpy as np
from tensorflow import Variable
from tensorflow.keras import Model, layers, optimizers
from tqdm import trange
from vrad import models
from vrad.inference import initializers
from vrad.inference.losses import ModelOutputLoss
from vrad.models.layers import (
    InferenceRNNLayers,
    LogLikelihoodLayer,
    MeansCovsLayer,
    MixMeansCovsLayer,
    ModelRNNLayers,
    NormalizationLayer,
    NormalKLDivergenceLayer,
    SampleNormalDistributionLayer,
    ThetaActivationLayer,
)
from vrad.utils.misc import check_arguments, replace_argument

_logger = logging.getLogger("VRAD")


class RIGO(models.GO):
    """RNN Inference/model network and Gaussian Observations (RIGO).

    Parameters
    ----------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    learn_covariances : bool
        Should we learn the covariance matrix for each state?
    rnn_type : str
        RNN to use, either 'lstm' or 'gru'.
    rnn_normalization : str
        Type of normalization to use in the inference network and generative model.
        Either 'layer', 'batch' or None.
    n_layers_inference : int
        Number of layers in the inference network.
    n_layers_model : int
        Number of layers in the generative model neural network.
    n_units_inference : int
        Number of units/neurons in the inference network.
    n_units_model : int
        Number of units/neurons in the generative model neural network.
    dropout_rate_inference : float
        Dropout rate in the inference network.
    dropout_rate_model : float
        Dropout rate in the generative model neural network.
    theta_normalization : str
        Type of normalization to apply to the posterior samples, theta.
        Either 'layer', 'batch' or None.
    alpha_xform : str
        Functional form of alpha. Either 'gumbel-softmax', 'softmax', 'softplus' or
        'relu'.
    alpha_temperature : float
        Temperature parameter for when alpha_xform = 'softmax' or 'gumbel-softmax'.
    learn_alpha_scaling : bool
        Should we learn a scaling for alpha?
    normalize_covariances : bool
        Should we trace normalize the state covariances?
    do_annealing : bool
        Should we use KL annealing during training?
    annealing_sharpness : float
        Parameter to control the annealing curve.
    n_epochs_annealing : int
        Number of epochs to perform annealing.
    learning_rate : float
        Learning rate for updating model parameters/weights.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    initial_covariances : np.ndarray
        Initial values for the state covariances. Should have shape (n_states,
        n_channels, n_channels).
    """

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        sequence_length: int,
        learn_covariances: bool,
        rnn_type: str,
        rnn_normalization: str,
        n_layers_inference: int,
        n_layers_model: int,
        n_units_inference: int,
        n_units_model: int,
        dropout_rate_inference: float,
        dropout_rate_model: float,
        theta_normalization: str,
        alpha_xform: str,
        alpha_temperature: float,
        learn_alpha_scaling: bool,
        normalize_covariances: bool,
        do_annealing: bool,
        annealing_sharpness: float,
        n_epochs_annealing: int,
        learning_rate: float,
        multi_gpu: bool = False,
        strategy: str = None,
        initial_covariances: np.ndarray = None,
    ):
        # Validation
        if n_layers_inference < 1 or n_layers_model < 1:
            raise ValueError("n_layers must be greater than zero.")

        if n_units_inference < 1 or n_units_model < 1:
            raise ValueError("n_units must be greater than zero.")

        if dropout_rate_inference < 0 or dropout_rate_model < 0:
            raise ValueError("dropout_rate must be greater than or equal to zero.")

        if alpha_xform not in ["gumbel-softmax", "softmax", "softplus", "relu"]:
            raise ValueError(
                "alpha_xform must be 'gumbel-softmax', 'softmax', 'softplus' or 'relu'."
            )

        if annealing_sharpness <= 0:
            raise ValueError("annealing_sharpness must be greater than zero.")

        if n_epochs_annealing < 0:
            raise ValueError(
                "n_epochs_annealing must be equal to or greater than zero."
            )

        # RNN and inference hyperparameters
        self.rnn_type = rnn_type
        self.rnn_normalization = rnn_normalization
        self.n_layers_inference = n_layers_inference
        self.n_layers_model = n_layers_model
        self.n_units_inference = n_units_inference
        self.n_units_model = n_units_model
        self.dropout_rate_inference = dropout_rate_inference
        self.dropout_rate_model = dropout_rate_model
        self.theta_normalization = theta_normalization
        self.alpha_xform = alpha_xform
        self.alpha_temperature = alpha_temperature

        # KL annealing
        self.do_annealing = do_annealing
        self.annealing_factor = Variable(0.0) if do_annealing else Variable(1.0)
        self.annealing_sharpness = annealing_sharpness
        self.n_epochs_annealing = n_epochs_annealing

        # Initialise the observation model
        # This will inherit the base model, build and compile the model
        super().__init__(
            n_states=n_states,
            n_channels=n_channels,
            sequence_length=sequence_length,
            learn_alpha_scaling=learn_alpha_scaling,
            normalize_covariances=normalize_covariances,
            learning_rate=learning_rate,
            multi_gpu=multi_gpu,
            strategy=strategy,
            initial_covariances=initial_covariances,
            learn_covariances=learn_covariances,
        )

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(
            n_states=self.n_states,
            n_channels=self.n_channels,
            sequence_length=self.sequence_length,
            rnn_type=self.rnn_type,
            rnn_normalization=self.rnn_normalization,
            n_layers_inference=self.n_layers_inference,
            n_layers_model=self.n_layers_model,
            n_units_inference=self.n_units_inference,
            n_units_model=self.n_units_model,
            dropout_rate_inference=self.dropout_rate_inference,
            dropout_rate_model=self.dropout_rate_model,
            theta_normalization=self.theta_normalization,
            alpha_xform=self.alpha_xform,
            alpha_temperature=self.alpha_temperature,
            initial_means=None,
            initial_covariances=self.initial_covariances,
            learn_means=False,
            learn_covariances=self.learn_covariances,
            learn_alpha_scaling=self.learn_alpha_scaling,
            normalize_covariances=self.normalize_covariances,
        )

    def compile(self):
        """Wrapper for the standard keras compile method.

        Sets up the optimizer and loss functions.
        """
        # Setup optimizer
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        # Loss functions
        ll_loss = ModelOutputLoss()
        kl_loss = ModelOutputLoss(self.annealing_factor)
        loss = [ll_loss, kl_loss]

        # Compile
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(
        self,
        *args,
        no_annealing_callback=False,
        use_tqdm=False,
        tqdm_class=None,
        use_tensorboard=None,
        tensorboard_dir=None,
        save_best_after=None,
        save_filepath=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

        Parameters
        ----------
        no_annealing_callback : bool
            Should we NOT update the annealing factor during training?
        use_tqdm : bool
            Should we use a tqdm progress bar instead of the usual output from
            tensorflow.
        tqdm_class : tqdm
            Class for the tqdm progress bar.
        use_tensorboard : bool
            Should we use TensorBoard?
        tensorboard_dir : str
            Path to the location to save the TensorBoard log files.
        save_best_after : int
            Epoch number after which we should save the best model. The best model is
            that which achieves the lowest loss.
        save_filepath : str
            Path to save the best model to.

        Returns
        -------
        history
            The training history.
        """
        if use_tqdm:
            args, kwargs = replace_argument(self.model.fit, "verbose", 0, args, kwargs)

        args, kwargs = replace_argument(
            func=self.model.fit,
            name="callbacks",
            item=self.create_callbacks(
                no_annealing_callback,
                use_tqdm,
                tqdm_class,
                use_tensorboard,
                tensorboard_dir,
                save_best_after,
                save_filepath,
            ),
            args=args,
            kwargs=kwargs,
            append=True,
        )

        return self.model.fit(*args, **kwargs)

    def reset_weight(self):
        """Reset the model as if you've built a new model.

        Resets the model weights, optimizer and annealing factor.
        """
        self.compile()
        initializers.reinitialize_model_weights(self.model)
        if self.do_annealing:
            self.annealing_factor.assign(0.0)

    def predict(self, *args, **kwargs) -> dict:
        """Wrapper for the standard keras predict method.

        Returns
        -------
        dict
            Dictionary with labels for each prediction.
        """
        predictions = self.model.predict(*args, *kwargs)
        return_names = ["ll_loss", "kl_loss", "alpha"]
        predictions_dict = dict(zip(return_names, predictions))
        return predictions_dict

    def predict_states(
        self, inputs, *args, concatenate: bool = False, **kwargs
    ) -> Union[list, np.ndarray]:
        """State mixing factors, alpha.

        Parameters
        ----------
        inputs : tensorflow.data.Dataset
            Prediction dataset.
        concatenate : bool
            Should we concatenate alpha for each subject? Optional, default
            is False.

        Returns
        -------
        np.ndarray
            State mixing factors with shape (n_subjects, n_samples, n_states) or
            (n_samples, n_states).
        """
        inputs = self._make_dataset(inputs)
        outputs = []
        for dataset in inputs:
            alpha = self.predict(dataset, *args, **kwargs)["alpha"]
            alpha = np.concatenate(alpha)
            outputs.append(alpha)
        if len(outputs) == 1:
            outputs = outputs[0]
        elif concatenate:
            outputs = np.concatenate(outputs)
        return outputs

    def losses(self, dataset, return_mean: bool = False) -> Tuple[float, float]:
        """Calculates the log-likelihood and KL loss for a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate losses for.
        return_mean : bool
            Should we return the mean loss over batches? Otherwise we return
            the sum. Optional, default is False.

        Returns
        -------
        ll_loss : float
            Negative log-likelihood loss.
        kl_loss : float
            KL divergence loss.
        """
        if return_mean:
            mean_or_sum = np.mean
        else:
            mean_or_sum = np.sum
        if isinstance(dataset, list):
            predictions = [self.predict(subject) for subject in dataset]
            ll_loss = mean_or_sum([mean_or_sum(p["ll_loss"]) for p in predictions])
            kl_loss = mean_or_sum([mean_or_sum(p["kl_loss"]) for p in predictions])
        else:
            predictions = self.predict(dataset)
            ll_loss = mean_or_sum(predictions["ll_loss"])
            kl_loss = mean_or_sum(predictions["kl_loss"])
        return ll_loss, kl_loss

    def free_energy(self, dataset, return_mean: bool = False) -> float:
        """Calculates the variational free energy of a dataset.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            Dataset to calculate the variational free energy for.
        return_mean : bool
            Should we return the mean free energy over batches? Otherwise
            we return the sum. Optional, default is False.

        Returns
        -------
        float
            Variational free energy for the dataset.
        """
        ll_loss, kl_loss = self.losses(dataset, return_mean=return_mean)
        free_energy = ll_loss + kl_loss
        return free_energy

    def burn_in(self, *args, **kwargs):
        """Burn-in training phase.

        Fits the model with means and covariances non-trainable.
        """
        if check_arguments(args, kwargs, 3, "epochs", 1, lt):
            _logger.warning(
                "Number of burn-in epochs is less than 1. Skipping burn-in."
            )
            return

        # Make means and covariances non-trainable and compile
        means_covs_layer = self.model.get_layer("means_covs")
        means_covs_layer.trainable = False
        self.compile()

        # Train the model
        self.fit(*args, **kwargs)

        # Make means and covariances trainable again and compile
        means_covs_layer.trainable = True
        self.compile()

    def sample_alpha(self, n_samples: int) -> np.ndarray:
        """Uses the model RNN to sample state mixing factors, alpha.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.

        Returns
        -------
        np.ndarray
            Sampled alpha.
        """
        # Get layers
        model_rnn_layer = self.model.get_layer("mod_rnn")
        mod_mu_layer = self.model.get_layer("mod_mu")
        log_mod_sigma_layer = self.model.get_layer("log_mod_sigma")
        theta_norm_layer = self.model.get_layer("theta_norm")
        alpha_layer = self.model.get_layer("alpha")

        # Sequence of the underlying logits theta
        theta_norm = np.zeros([self.sequence_length, self.n_states], dtype=np.float32)

        # Normally distributed random numbers used to sample the logits theta
        epsilon = np.random.normal(0, 1, [n_samples + 1, self.n_states]).astype(
            np.float32
        )

        # Activate the first state for the first sample
        theta_norm[-1, 0] = 1

        # Sample the state fixing factors
        alpha = np.empty([n_samples, self.n_states], dtype=np.float32)
        for i in trange(n_samples, desc="Sampling state time course", ncols=98):

            # If there are leading zeros we trim theta so that we don't pass the zeros
            trimmed_theta = theta_norm[~np.all(theta_norm == 0, axis=1)][
                np.newaxis, :, :
            ]

            # Predict the probability distribution function for theta one time step
            # in the future,
            # p(theta|theta_<t) ~ N(mod_mu, sigma_theta_jt)
            model_rnn = model_rnn_layer(trimmed_theta)
            mod_mu = mod_mu_layer(model_rnn)[0, -1]
            log_mod_sigma = log_mod_sigma_layer(model_rnn)[0, -1]
            mod_sigma = np.exp(log_mod_sigma)

            # Shift theta one time step to the left
            theta_norm = np.roll(theta_norm, -1, axis=0)

            # Sample from the probability distribution function
            theta = mod_mu + mod_sigma * epsilon[i]
            theta_norm[-1] = theta_norm_layer(theta[np.newaxis, np.newaxis, :])[0]

            # Calculate the state mixing factors
            alpha[i] = alpha_layer(theta_norm[-1][np.newaxis, np.newaxis, :])[0, 0]

        return alpha


def _model_structure(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    rnn_type: str,
    rnn_normalization: str,
    n_layers_inference: int,
    n_layers_model: int,
    n_units_inference: int,
    n_units_model: int,
    dropout_rate_inference: float,
    dropout_rate_model: float,
    theta_normalization: str,
    alpha_xform: str,
    alpha_temperature: float,
    initial_means: np.ndarray,
    initial_covariances: np.ndarray,
    learn_means: bool,
    learn_covariances: bool,
    learn_alpha_scaling: bool,
    normalize_covariances: bool,
):
    """Model structure.

    Parameters
    ----------
    n_states : int
        Numeber of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    rnn_type : int
        RNN to use, either 'lstm' or 'gru'.
    rnn_normalization : str
        Type of normalization to use in the inference network and generative model.
        Either 'layer', 'batch' or None.
    n_layers_inference : int
        Number of layers in the inference network.
    n_layers_model : int
        Number of layers in the generative model neural network.
    n_units_inference : int
        Number of units/neurons in the inference network.
    n_units_model : int
        Number of units/neurons in the generative model neural network.
    dropout_rate_inference : float
        Dropout rate in the inference network.
    dropout_rate_model : float
        Dropout rate in the generative model neural network.
    theta_normalization : str
        Type of normalization to apply to the posterior samples, theta.
        Either 'layer', 'batch' or None.
    learn_means : bool
        Should we learn the mean vector for each state?
    learn_covariances : bool
        Should we learn the covariance matrix for each state?
    initial_means : np.ndarray
        Initial values for the state means. Should have shape (n_states, n_channels).
    initial_covariances : np.ndarray
        Initial values for the state covariances. Should have shape (n_states,
        n_channels, n_channels).
    alpha_xform : str
        Functional form of alpha. Either 'gumbel-softmax', 'softmax', 'softplus' or
        'relu'.
    alpha_temperature : float
        Temperature parameter for when alpha_xform = 'softmax' or 'gumbel-softmax'.
    learn_alpha_scaling : bool
        Should we learn a scaling for alpha?
    normalize_covariances : bool
        Should we trace normalize the state covariances?

    Returns
    -------
    tensorflow.keras.Model
        Keras model built using the functional API.
    """

    # Layer for input
    inputs = layers.Input(shape=(sequence_length, n_channels), name="data")

    # Inference RNN:
    # - Learns q(theta) ~ N(theta | inf_mu, inf_sigma), where
    #     - inf_mu        ~ affine(RNN(inputs_<=t))
    #     - log_inf_sigma ~ affine(RNN(inputs_<=t))

    # Definition of layers
    inference_input_dropout_layer = layers.Dropout(
        dropout_rate_inference, name="data_drop"
    )
    inference_output_layers = InferenceRNNLayers(
        rnn_type,
        rnn_normalization,
        n_layers_inference,
        n_units_inference,
        dropout_rate_inference,
        name="inf_rnn",
    )
    inf_mu_layer = layers.Dense(n_states, name="inf_mu")
    log_inf_sigma_layer = layers.Dense(n_states, name="log_inf_sigma")

    # Layers to sample theta from q(theta) and to convert to state mixing
    # factors alpha
    theta_layer = SampleNormalDistributionLayer(name="theta")
    theta_norm_layer = NormalizationLayer(theta_normalization, name="theta_norm")
    alpha_layer = ThetaActivationLayer(alpha_xform, alpha_temperature, name="alpha")

    # Data flow
    inference_input_dropout = inference_input_dropout_layer(inputs)
    inference_output = inference_output_layers(inference_input_dropout)
    inf_mu = inf_mu_layer(inference_output)
    log_inf_sigma = log_inf_sigma_layer(inference_output)
    theta = theta_layer([inf_mu, log_inf_sigma])
    theta_norm = theta_norm_layer(theta)
    alpha = alpha_layer(theta_norm)

    # Observation model:
    # - We use a multivariate normal with a mean vector and covariance matrix for
    #   each state as the observation model.
    # - We calculate the likelihood of generating the training data with alpha
    #   and the observation model.

    # Definition of layers
    means_covs_layer = MeansCovsLayer(
        n_states,
        n_channels,
        learn_means=learn_means,
        learn_covariances=learn_covariances,
        normalize_covariances=normalize_covariances,
        initial_means=initial_means,
        initial_covariances=initial_covariances,
        name="means_covs",
    )
    mix_means_covs_layer = MixMeansCovsLayer(
        n_states, n_channels, learn_alpha_scaling, name="mix_means_covs"
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    mu, D = means_covs_layer(inputs)  # inputs not used
    m, C = mix_means_covs_layer([alpha, mu, D])
    ll_loss = ll_loss_layer([inputs, m, C])

    # Model RNN:
    # - Learns p(theta_t |theta_<t) ~ N(theta_t | mod_mu, mod_sigma), where
    #     - mod_mu        ~ affine(RNN(theta_<t))
    #     - log_mod_sigma ~ affine(RNN(theta_<t))

    # Definition of layers
    model_input_dropout_layer = layers.Dropout(
        dropout_rate_model, name="theta_norm_drop"
    )
    model_output_layers = ModelRNNLayers(
        rnn_type,
        rnn_normalization,
        n_layers_model,
        n_units_model,
        dropout_rate_model,
        name="mod_rnn",
    )
    mod_mu_layer = layers.Dense(n_states, name="mod_mu")
    log_mod_sigma_layer = layers.Dense(n_states, name="log_mod_sigma")
    kl_loss_layer = NormalKLDivergenceLayer(name="kl")

    # Data flow
    model_input_dropout = model_input_dropout_layer(theta_norm)
    model_output = model_output_layers(model_input_dropout)
    mod_mu = mod_mu_layer(model_output)
    log_mod_sigma = log_mod_sigma_layer(model_output)
    kl_loss = kl_loss_layer([inf_mu, log_inf_sigma, mod_mu, log_mod_sigma])

    return Model(inputs=inputs, outputs=[ll_loss, kl_loss, alpha])
