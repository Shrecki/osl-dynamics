import numpy as np
from tensorflow.keras import Model, layers, optimizers
from tensorflow.python import Variable, zeros
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from vrad.inference.callbacks import AnnealingCallback, BurninCallback
from vrad.inference.layers import (
    MVNLayer,
    TrainableVariablesLayer,
    ReparameterizationLayer,
    KLDivergenceLayer,
    LogLikelihoodLayer,
)
from vrad.utils.misc import listify


def create_model(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    learn_means: bool,
    learn_covs: bool,
    initial_mean: np.ndarray,
    initial_cholesky_cov: np.ndarray,
    n_units_encoder: int = 64,
    n_units_decoder: int = 64,
    dropout_rate_encoder: float = 0.3,
    dropout_rate_decoder: float = 0.3,
    activation_function: str = "softmax",
    do_annealing: bool = True,
    annealing_sharpness: float = 5.0,
    n_epochs_annealing: int = 80,
    n_epochs_burnin: int = 10,
    learning_rate: float = 0.01,
    clip_normalization: float = None,
    multi_gpu: bool = False,
):
    annealing_factor = Variable(0.0) if do_annealing else Variable(1.0)

    # Setup optimizer
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, clipnorm=clip_normalization,
    )

    # Stretegy for distributed learning
    if multi_gpu:
        strategy = MirroredStrategy()
    else:
        strategy = get_strategy()

    # Compile the model
    with strategy.scope():
        model = _model_structure(
            n_states=n_states,
            n_channels=n_channels,
            sequence_length=sequence_length,
            n_units_encoder=n_units_encoder,
            n_units_decoder=n_units_decoder,
            dropout_rate_encoder=dropout_rate_encoder,
            dropout_rate_decoder=dropout_rate_decoder,
            learn_means=learn_means,
            learn_covs=learn_covs,
            initial_mean=initial_mean,
            initial_cholesky_cov=initial_cholesky_cov,
            activation_function=activation_function,
        )

        model.compile(
            optimizer=optimizer,
            loss=[_ll_loss, _kl_loss(annealing_factor=annealing_factor)],
        )

    # Callbacks
    burnin_callback = BurninCallback(epochs=n_epochs_burnin)

    annealing_callback = AnnealingCallback(
        annealing_factor=annealing_factor,
        annealing_sharpness=annealing_sharpness,
        n_epochs_annealing=n_epochs_annealing,
    )

    model.original_fit_method = model.fit

    def anneal_fit(*args, **kwargs):
        args = list(args)
        if len(args) > 5:
            args[5] = listify(args[5]) + [annealing_callback, burnin_callback]
        if "callbacks" in kwargs:
            kwargs["callbacks"] = listify(kwargs["callbacks"]) + [
                annealing_callback,
                burnin_callback,
            ]

        return model.original_fit_method(*args, **kwargs)

    model.fit = anneal_fit

    return model


def _ll_loss(y_true, ll_loss):
    """The first output of the model is the negative log likelihood
       so we just need to return it."""
    return ll_loss


def _kl_loss(annealing_factor):
    def kl_loss_fn(y_true, kl_loss):
        """Second output of the model is the KL divergence loss.
        We multiply with an annealing factor."""
        return annealing_factor * kl_loss

    return kl_loss_fn


def _model_structure(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    n_units_encoder: int,
    n_units_decoder: int,
    dropout_rate_encoder: float,
    dropout_rate_decoder: float,
    learn_means: bool,
    learn_covs: bool,
    initial_mean: np.ndarray,
    initial_cholesky_cov: np.ndarray,
    activation_function: str = "softmax",
):
    # Layer for input
    inputs = layers.Input(shape=(sequence_length, n_channels))

    # Inference RNN (encoder):
    # - q(theta_t) ~ N(m_theta_t, s2_theta_t)
    # - m_theta_t  ~ affine(RNN(Y_<=t))
    # - s2_theta_t ~ softplus(RNN(Y_<=t))

    # Definition of layers
    input_normalisation_layer = layers.LayerNormalization()
    inference_input_dropout_layer = layers.Dropout(dropout_rate_encoder)
    inference_output_layer = layers.Bidirectional(
        layer=layers.LSTM(n_units_encoder, return_sequences=True, stateful=False)
    )
    inference_normalisation_layer = layers.LayerNormalization()
    inference_output_dropout_layer = layers.Dropout(dropout_rate_encoder)
    m_theta_t_layer = layers.Dense(n_states, activation="linear")
    s2_theta_t_layer = layers.Dense(n_states, activation="softplus")

    # Layer to generate a sample from q(theta_t) ~ N(m_theta_t, s2_theta_t) via the
    # reparameterisation trick
    theta_t_layer = ReparameterizationLayer()

    # Inference RNN data flow
    inputs_norm = input_normalisation_layer(inputs)
    inputs_dropout = inference_input_dropout_layer(inputs_norm)
    inference_output = inference_output_layer(inputs_dropout)
    inference_output_norm = inference_normalisation_layer(inference_output)
    inference_output_dropout = inference_output_dropout_layer(inference_output_norm)
    m_theta_t = m_theta_t_layer(inference_output_dropout)
    s2_theta_t = s2_theta_t_layer(inference_output_dropout)
    theta_t = theta_t_layer([m_theta_t, s2_theta_t])

    # Model RNN (decoder):
    # - p(theta_t|theta_<t) ~ N(mu_theta_jt, sigma2_theta_j)
    # - mu_theta_jt         ~ affine(RNN(theta_<t))
    # - sigma2_theta_j      = trainable constant

    # Definition of layers
    model_input_dropout_layer = layers.Dropout(dropout_rate_decoder)
    model_output_layer = layers.LSTM(
        n_units_decoder, return_sequences=True, stateful=False
    )
    model_normalisation_layer = layers.LayerNormalization()
    model_output_dropout_layer = layers.Dropout(dropout_rate_decoder)
    mu_theta_jt_layer = layers.Dense(n_states, activation="linear")
    sigma2_theta_j_layer = TrainableVariablesLayer(
        [n_states], initial_values=zeros(n_states), trainable=True
    )

    # Layers for the means and covariances for observation model of each state
    observation_means_covs_layer = MVNLayer(
        n_states,
        n_channels,
        learn_means=learn_means,
        learn_covs=learn_covs,
        initial_means=initial_mean,
        initial_pseudo_sigmas=initial_cholesky_cov,
    )

    # Layers to calculate loss as the free energy = LL + KL
    log_likelihood_layer = LogLikelihoodLayer(
        n_states, n_channels, alpha_xform=activation_function, name="ll"
    )
    kl_loss_layer = KLDivergenceLayer(n_states, n_channels, name="kl")

    # Model RNN data flow
    model_input_dropout = model_input_dropout_layer(theta_t)
    model_output = model_output_layer(model_input_dropout)
    model_output_norm = model_normalisation_layer(model_output)
    model_output_dropout = model_output_dropout_layer(model_output_norm)
    mu_theta_jt = mu_theta_jt_layer(model_output_dropout)
    sigma2_theta_j = sigma2_theta_j_layer(inputs)  # inputs not used
    mu_j, D_j = observation_means_covs_layer(inputs)  # inputs not used
    ll_loss = log_likelihood_layer([inputs, theta_t, mu_j, D_j])
    kl_loss = kl_loss_layer([m_theta_t, s2_theta_t, mu_theta_jt, sigma2_theta_j])

    outputs = [
        ll_loss,
        kl_loss,
        theta_t,
        m_theta_t,
        s2_theta_t,
        mu_theta_jt,
        sigma2_theta_j,
        mu_j,
        D_j,
    ]

    model = Model(inputs=inputs, outputs=outputs,)

    return model
