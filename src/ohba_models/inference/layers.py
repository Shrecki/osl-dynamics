"""Custom Tensorflow layers used in the inference network and generative model.

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import activations, layers, regularizers
from ohba_models.inference.initializers import WeightInitializer

tfb = tfp.bijectors


def NormalizationLayer(norm_type: str, *args, **kwargs):
    """Returns a normalization layer.

    Parameters
    ----------
    norm_type : str
        Type of normalization layer. Either 'layer', 'batch' or None.
    """
    if norm_type == "layer":
        return layers.LayerNormalization(*args, **kwargs)
    elif norm_type == "batch":
        return layers.BatchNormalization(*args, **kwargs)
    elif norm_type is None:
        return DummyLayer(*args, **kwargs)
    else:
        raise NotImplementedError(norm_type)


def RNNLayer(rnn_type: str, *args, **kwargs):
    """Returns an RNN layer.

    Parameters
    ----------
    rnn_type : str
        Type of RNN. Either 'lstm' or 'gru'.
    """
    if rnn_type == "lstm":
        return layers.LSTM(*args, **kwargs)
    elif rnn_type == "gru":
        return layers.GRU(*args, **kwargs)
    else:
        raise NotImplementedError(rnn_type)


class DummyLayer(layers.Layer):
    """Dummy layer.

    Returns the inputs without modification.
    """

    def call(self, inputs, **kwargs):
        return inputs


class ConcatenateLayer(layers.Layer):
    """Concatenates a set of tensors.

    Wrapper for tf.concat().

    Parameters
    ----------
    axis : int
        Axis to concatenate along.
    """

    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.concat(inputs, axis=self.axis)


class MatMulLayer(layers.Layer):
    """Multiplies a set of matrices.

    Wrapper for tf.matmul().
    """

    def call(self, inputs, **kwargs):
        # If [A, B, C] is passed, we return matmul(A, matmul(B, C))
        out = inputs[-1]
        for tensor in inputs[len(inputs) - 2 :: -1]:
            out = tf.matmul(tensor, out)
        return out


class TFRangeLayer(layers.Layer):
    """Wrapper for tf.range"""

    def __init__(self, limit, **kwargs):
        super().__init__(**kwargs)
        self.limit = limit

    def call(self, inputs):
        return tf.range(self.limit)


class SampleNormalDistributionLayer(layers.Layer):
    """Layer for sampling from a normal distribution.

    This layer accepts the mean and the standard deviation and
    outputs samples from a normal distribution.
    """

    def call(self, inputs, training=None, **kwargs):
        mu, sigma = inputs
        if training:
            N = tfp.distributions.Normal(loc=mu, scale=sigma)
            return N.sample()
        else:
            return mu


class SampleGumbelSoftmaxDistributionLayer(layers.Layer):
    """Layer for sampling from a Gumbel-Softmax distribution."""

    def call(self, inputs, **kwargs):
        gs = tfp.distributions.RelaxedOneHotCategorical(temperature=0.5, logits=inputs)
        return gs.sample()


class SoftmaxLayer(layers.Layer):
    """Layer for applying a softmax activation function.

    Parameters
    ----------
    initial_temperature : float
        Temperature parameter.
    learn_temperature : bool
        Should we learn the alpha temperature?
    """

    def __init__(
        self,
        initial_temperature: float,
        learn_temperature: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.initial_temperature = initial_temperature
        self.learn_temperature = learn_temperature
        self.temperature_initializer = WeightInitializer(self.initial_temperature)

    def build(self, input_shape):
        self.temperature = self.add_weight(
            "temperature",
            shape=(),
            dtype=tf.float32,
            initializer=self.temperature_initializer,
            trainable=self.learn_temperature,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        return activations.softmax(inputs / self.temperature, axis=2)


class MeanVectorsLayer(layers.Layer):
    """Layer to learn a set of mean vectors.

    The vectors are free parameters.

    Parameters
    ----------
    n : int
        Number of vectors.
    m : int
        Number of elements.
    learn : bool
        Should we learn the vectors?
    initial_value : np.ndarray
        Initial value for the vectors.
    """

    def __init__(
        self,
        n: int,
        m: int,
        learn: bool,
        initial_value: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.m = m
        self.learn = learn
        if initial_value is None:
            self.initial_value = np.zeros([n, m], dtype=np.float32)
        else:
            if initial_value.ndim != 2:
                raise ValueError(
                    "a (n_modes, n_channels) array must be passed for initial_means."
                )
            if initial_value.shape[-1] != m:
                raise ValueError(
                    f"mismatch between the number of channels ({m}) and number of "
                    + f"elements in initial_means ({initial_value.shape[-1]})."
                )
            if initial_value.shape[0] != n:
                raise ValueError(
                    f"mismatch bettwen the number of modes and vectors in "
                    + f"initial_means ({initial_value.shape[0]})."
                )
            self.initial_value = initial_value.astype("float32")
        self.vectors_initializer = WeightInitializer(self.initial_value)

    def build(self, input_shape):
        self.vectors = self.add_weight(
            "vectors",
            shape=(self.n, self.m),
            dtype=tf.float32,
            initializer=self.vectors_initializer,
            trainable=self.learn,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        return self.vectors


class CovarianceMatricesLayer(layers.Layer):
    """Layer to learn a set of covariance matrices.

    A cholesky factor is learnt and used to calculate a covariance matrix as
    C = LL^T, where L is the cholesky factor. The cholesky factor is learnt as
    a vector of free parameters.

    Parameters
    ----------
    n : int
        Number of matrices.
    m : int
        Number of rows/columns.
    learn : bool
        Should the matrices be learnable?
    initial_value : np.ndarray
        Initial values for the matrices.
    """

    def __init__(
        self,
        n: int,
        m: int,
        learn: bool,
        initial_value: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.m = m
        self.learn = learn

        # Bijector used to transform learnable vectors to covariance matrices
        self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])

        # Initialisation of matrices
        if initial_value is None:
            self.initial_value = np.stack([np.eye(m, dtype=np.float32)] * n)
        else:
            if initial_value.ndim != 3:
                raise ValueError(
                    "a (n_modes, n_channels, n_channels) array must be passed for "
                    + "initial_covariances."
                )
            if initial_value.shape[-1] != m:
                raise ValueError(
                    f"mismatch between the number of channels ({m}) and number of "
                    + "rows/columns in initial_covariances "
                    + f"({initial_value.shape[-1]})."
                )
            if initial_value.shape[0] != n:
                raise ValueError(
                    f"mismatch bettwen the number of modes and matrices in "
                    + f"initial_covariances ({initial_value.shape[0]})."
                )
            self.initial_value = initial_value.astype("float32")
        self.initial_flattened_cholesky_factors = self.bijector.inverse(
            self.initial_value
        )
        self.flattened_cholesky_factors_initializer = WeightInitializer(
            self.initial_flattened_cholesky_factors
        )

    def build(self, input_shape):
        self.flattened_cholesky_factors = self.add_weight(
            "flattened_cholesky_factors",
            shape=(self.n, self.m * (self.m + 1) // 2),
            dtype=tf.float32,
            initializer=self.flattened_cholesky_factors_initializer,
            trainable=self.learn,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        return self.bijector(self.flattened_cholesky_factors)


class CorrelationMatricesLayer(layers.Layer):
    """Layer to learn a set of correlation matrices.

    A cholesky factor is learnt as a vector of free parameters and used to
    calculate a correlation matrix.

    Parameters
    ----------
    n : int
        Number of matrices.
    m : int
        Number of rows/columns.
    learn : bool
        Should the matrices be learnable?
    initial_value : np.ndarray
        Initial values for the matrices.
    """

    def __init__(
        self,
        n: int,
        m: int,
        learn: bool,
        initial_value: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.m = m
        self.learn = learn

        # Bijector used to transform learnable vectors to correlation matrices
        self.bijector = tfb.Chain(
            [tfb.CholeskyOuterProduct(), tfb.CorrelationCholesky()]
        )

        # Initialisation of matrices
        if initial_value is None:
            self.initial_value = np.stack([np.eye(m, dtype=np.float32)] * n)
        else:
            if initial_value.ndim != 3:
                raise ValueError(
                    "a (n_modes, n_channels, n_channels) array must be passed for "
                    + "initial_fcs."
                )
            if initial_value.shape[-1] != m:
                raise ValueError(
                    f"mismatch between the number of channels ({m}) and number of "
                    + "rows/columns in initial_fcs "
                    + f"({initial_value.shape[-1]})."
                )
            if initial_value.shape[0] != n:
                raise ValueError(
                    f"mismatch bettwen the number of modes and matrices in "
                    + f"initial_fcs ({initial_value.shape[0]})."
                )
            self.initial_value = initial_value.astype("float32")
        self.initial_flattened_cholesky_factors = self.bijector.inverse(
            self.initial_value
        )
        self.flattened_cholesky_factors_initializer = WeightInitializer(
            self.initial_flattened_cholesky_factors
        )

    def build(self, input_shape):
        self.flattened_cholesky_factors = self.add_weight(
            "flattened_cholesky_factors",
            shape=(self.n, self.m * (self.m - 1) // 2),
            dtype=tf.float32,
            initializer=self.flattened_cholesky_factors_initializer,
            trainable=self.learn,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        return self.bijector(self.flattened_cholesky_factors)


class DiagonalMatricesLayer(layers.Layer):
    """Layer to learn a set of diagonal matrices.

    The diagonal is forced to be positive using a softplus transformation.

    Parameters
    ----------
    n : int
        Number of matrices.
    m : int
        Number of rows/columns.
    learn : bool
        Should the matrices be learnable?
    initial_value : np.ndarray
        Initial values for the matrices.
    """

    def __init__(
        self,
        n: int,
        m: int,
        learn: bool,
        initial_value: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.m = m
        self.learn = learn

        # Softplus transformation to ensure diagonal is positive
        self.bijector = tfb.Softplus()

        # Initialisation for the diagonals
        if initial_value is None:
            self.initial_value = np.ones([n, m], dtype=np.float32)
        else:
            if initial_value.ndim != 2:
                raise ValueError(
                    "a (n_modes, n_channels) array must be passed for initial_value."
                )
            if initial_value.shape[-1] != m:
                raise ValueError(
                    f"mismatch between the number of channels ({m}) and number of "
                    + f"elements in initial_value ({initial_value.shape[-1]})."
                )
            if initial_value.shape[0] != n:
                raise ValueError(
                    f"mismatch bettwen the number of modes and vectors in "
                    + f"initial_value ({initial_value.shape[0]})."
                )
            self.initial_value = initial_value.astype("float32")
        self.initial_diagonals = self.bijector.inverse(self.initial_value)
        self.diagonals_initializer = WeightInitializer(self.initial_diagonals)

    def build(self, input_shape):
        self.diagonals = self.add_weight(
            "diagonals",
            shape=(self.n, self.m),
            dtype=tf.float32,
            initializer=self.diagonals_initializer,
            trainable=self.learn,
        )
        self.built = True

    def call(self, alpha, **kwargs):
        D = self.bijector(self.diagonals)
        D = tf.linalg.diag(D)
        return D


class MixVectorsLayer(layers.Layer):
    """Mix a set of vectors.

    The mixture is calculated as m_t = Sum_j alpha_jt mu_j,
    where mu_j are the vectors and alpha_jt are mixing coefficients.
    """

    def call(self, inputs, **kwargs):

        # Unpack the inputs:
        # - alpha.shape = (None, sequence_length, n_modes)
        # - mu.shape    = (n_modes, n_channels)
        alpha, mu = inputs

        # Calculate the mixture: m_t = Sum_j alpha_jt mu_j
        alpha = tf.expand_dims(alpha, axis=-1)
        mu = tf.expand_dims(tf.expand_dims(mu, axis=0), axis=0)
        m = tf.reduce_sum(tf.multiply(alpha, mu), axis=2)

        return m


class MixMatricesLayer(layers.Layer):
    """Layer to mix matrices.

    The mixture is calculated as C_t = Sum_j alpha_jt D_j,
    where D_j are the matrices and alpha_jt are mixing coefficients.
    """

    def call(self, inputs, **kwargs):

        # Unpack the inputs:
        # - alpha.shape = (None, sequence_length, n_modes)
        # - D.shape     = (n_modes, n_channels, n_channels)
        alpha, D = inputs

        # Calculate the mixture: C_t = Sum_j alpha_jt D_j
        alpha = tf.expand_dims(tf.expand_dims(alpha, axis=-1), axis=-1)
        D = tf.expand_dims(tf.expand_dims(D, axis=0), axis=0)
        C = tf.reduce_sum(tf.multiply(alpha, D), axis=2)

        return C


class LogLikelihoodLossLayer(layers.Layer):
    """Layer to calculate the negative log likelihood.

    The negative log-likelihood is calculated assuming a multivariate normal
    probability density and its value is added to the loss function.
    """

    def call(self, inputs):
        x, mu, sigma = inputs

        # Multivariate normal distribution
        mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mu,
            scale_tril=tf.linalg.cholesky(sigma),
            allow_nan_stats=False,
        )

        # Calculate the log-likelihood
        ll_loss = mvn.log_prob(x)

        # Sum over time dimension and average over the batch dimension
        ll_loss = tf.reduce_sum(ll_loss, axis=1)
        ll_loss = tf.reduce_mean(ll_loss, axis=0)

        # Add the negative log-likelihood to the loss
        nll_loss = -ll_loss
        self.add_loss(nll_loss)
        self.add_metric(nll_loss, name=self.name)

        return nll_loss


class KLDivergenceLayer(layers.Layer):
    """Layer to calculate a KL divergence between two Normal distributions.

    Parameters
    ----------
    clip_start : int
        Index to clip the sequences inputted to this layer.
    """

    def __init__(self, clip_start: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.clip_start = clip_start

    def call(self, inputs, **kwargs):
        inference_mu, inference_sigma, model_mu, model_sigma = inputs

        # The model network predicts one time step into the future compared to
        # the inference network. We clip the sequences to ensure we are comparing
        # the correct time points.
        model_mu = model_mu[:, self.clip_start : -1]
        model_sigma = model_sigma[:, self.clip_start : -1]

        inference_mu = inference_mu[:, self.clip_start + 1 :]
        inference_sigma = inference_sigma[:, self.clip_start + 1 :]

        # Calculate the KL divergence between the posterior and prior
        prior = tfp.distributions.Normal(loc=model_mu, scale=model_sigma)
        posterior = tfp.distributions.Normal(loc=inference_mu, scale=inference_sigma)
        kl_loss = tfp.distributions.kl_divergence(
            posterior, prior, allow_nan_stats=False
        )

        # Sum the KL loss for each mode and time point and average over batches
        kl_loss = tf.reduce_sum(kl_loss, axis=2)
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = tf.reduce_mean(kl_loss, axis=0)

        return kl_loss


class KLLossLayer(layers.Layer):
    """Layer to calculate the KL loss.

    This layer sums KL divergences if multiple values as passed, applies an
    annealing factor and adds the value to the loss function.

    Parameters
    ----------
    do_annealing : bool
        Should we perform KL annealing?
    """

    def __init__(self, do_annealing: bool, **kwargs):
        super().__init__(**kwargs)
        if do_annealing:
            self.annealing_factor = tf.Variable(0.0, trainable=False)
        else:
            self.annealing_factor = tf.Variable(1.0, trainable=False)

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            # Sum KL divergences
            inputs = tf.add_n(inputs)

        # KL annealing
        kl_loss = tf.multiply(inputs, self.annealing_factor)

        # Add to loss
        self.add_loss(kl_loss)
        self.add_metric(kl_loss, name=self.name)

        return kl_loss


class InferenceRNNLayer(layers.Layer):
    """RNN inference network.

    Parameters
    ----------
    rnn_type : str
        Either 'lstm' or 'gru'.
    norm_type : str
        Either 'layer', 'batch' or None.
    act_type : 'str'
        Activation type, e.g. 'relu', 'elu', etc.
    n_layers : int
        Number of layers.
    n_units : int
        Number of units/neurons per layer.
    drop_rate : float
        Dropout rate for the output of each layer.
    """

    def __init__(
        self,
        rnn_type: str,
        norm_type: str,
        act_type: str,
        n_layers: int,
        n_units: int,
        drop_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                layers.Bidirectional(
                    layer=RNNLayer(
                        rnn_type, n_units, return_sequences=True, stateful=False
                    )
                )
            )
            self.layers.append(NormalizationLayer(norm_type))
            self.layers.append(layers.Activation(act_type))
            self.layers.append(layers.Dropout(drop_rate))

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs


class ModelRNNLayer(layers.Layer):
    """RNN generative model.

    Parameters
    ----------
    rnn_type : str
        Either 'lstm' or 'gru'.
    norm_type : str
        Either 'layer', 'batch' or None.
    act_type : 'str'
        Activation type, e.g. 'relu', 'elu', etc.
    n_layers : int
        Number of layers.
    n_units : int
        Number of units/neurons per layer.
    drop_rate : float
        Dropout rate for the output of each layer.
    """

    def __init__(
        self,
        rnn_type: str,
        norm_type: str,
        act_type: str,
        n_layers: int,
        n_units: int,
        drop_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = []
        for n in range(n_layers):
            self.layers.append(
                RNNLayer(rnn_type, n_units, return_sequences=True, stateful=False)
            )
            self.layers.append(NormalizationLayer(norm_type))
            self.layers.append(layers.Activation(act_type))
            self.layers.append(layers.Dropout(drop_rate))

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs


class WaveNetLayer(layers.Layer):
    """Layer for generating data using a WaveNet architecture.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    n_filters : int
        Number of filters for each convolution.
    n_layers : int
        Number of dilated causal convolution layers in each residual block.
    local_conditioning : bool
        Will we condition the WaveNet on a second input?
    """

    def __init__(
        self,
        n_channels: int,
        n_filters: int,
        n_layers: int,
        local_conditioning: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.local_conditioning = local_conditioning

        self.causal_conv_layer = layers.Conv1D(
            filters=n_filters,
            kernel_size=2,
            dilation_rate=1,
            padding="causal",
            use_bias=False,
            kernel_regularizer=regularizers.l2(),
        )
        self.residual_block_layers = []
        for i in range(1, n_layers):
            if local_conditioning:
                self.residual_block_layers.append(
                    LocallyConditionedWaveNetResidualBlockLayer(
                        filters=n_filters, dilation_rate=2**i
                    )
                )
            else:
                self.residual_block_layers.append(
                    WaveNetResidualBlockLayer(filters=n_filters, dilation_rate=2**i)
                )
        self.dense_layers = [
            layers.Conv1D(
                filters=n_channels,
                kernel_size=1,
                padding="same",
                kernel_regularizer=regularizers.l2(),
                bias_regularizer=regularizers.l2(),
            ),
            layers.Conv1D(
                filters=n_channels,
                kernel_size=1,
                padding="same",
                kernel_regularizer=regularizers.l2(),
                bias_regularizer=regularizers.l2(),
            ),
        ]

    def call(self, inputs, **kwargs):
        if self.local_conditioning:
            x, h = inputs
            out = self.causal_conv_layer(x)
        else:
            out = self.causal_conv_layer(inputs)
        skips = []
        for layer in self.residual_block_layers:
            if self.local_conditioning:
                out, skip = layer([out, h])
            else:
                out, skip = layer(out)
            skips.append(skip)
        out = tf.add_n(skips)
        for layer in self.dense_layers:
            out = activations.selu(out)
            out = layer(out)
        return out


class WaveNetResidualBlockLayer(layers.Layer):
    """Layer for a residual block in WaveNet.

    Parameters
    ----------
    filters : int
        Number of filters for the convolutions.
    dilation_rate : int
        Dilation rate for the causal convolutions.
    """

    def __init__(self, filters: int, dilation_rate: int, **kwargs):
        super().__init__(**kwargs)
        self.filter_layer = layers.Conv1D(
            filters,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding="causal",
            use_bias=False,
            kernel_regularizer=regularizers.l2(),
        )
        self.gate_layer = layers.Conv1D(
            filters,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding="causal",
            use_bias=False,
            kernel_regularizer=regularizers.l2(),
        )
        self.res_layer = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            kernel_regularizer=regularizers.l2(),
            bias_regularizer=regularizers.l2(),
        )
        self.skip_layer = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            kernel_regularizer=regularizers.l2(),
            bias_regularizer=regularizers.l2(),
        )

    def call(self, inputs, **kwargs):
        filter_ = self.filter_layer(inputs)
        gate = self.gate_layer(inputs)
        z = tf.tanh(filter_) * tf.sigmoid(gate)
        residual = self.res_layer(z)
        skip = self.skip_layer(z)
        return inputs + residual, skip


class LocallyConditionedWaveNetResidualBlockLayer(layers.Layer):
    """Layer for a residual block in WaveNet.

    Parameters
    ----------
    filters : int
        Number of filters for the convolutions.
    dilation_rate : int
        Dilation rate for the causal convolutions.
    """

    def __init__(self, filters: int, dilation_rate: int, **kwargs):
        super().__init__(**kwargs)
        self.h_transform_layer = layers.Conv1D(filters, kernel_size=1, padding="same")
        self.x_filter_layer = layers.Conv1D(
            filters,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding="causal",
            kernel_regularizer=regularizers.l2(),
            use_bias=False,
        )
        self.y_filter_layer = layers.Conv1D(
            filters,
            kernel_size=1,
            dilation_rate=dilation_rate,
            padding="same",
            kernel_regularizer=regularizers.l2(),
            use_bias=False,
        )
        self.x_gate_layer = layers.Conv1D(
            filters,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding="causal",
            kernel_regularizer=regularizers.l2(),
            use_bias=False,
        )
        self.y_gate_layer = layers.Conv1D(
            filters,
            kernel_size=1,
            dilation_rate=dilation_rate,
            padding="same",
            kernel_regularizer=regularizers.l2(),
            use_bias=False,
        )
        self.res_layer = layers.Conv1D(filters=filters, kernel_size=1, padding="same")
        self.skip_layer = layers.Conv1D(filters=filters, kernel_size=1, padding="same")

    def call(self, inputs, training=None, **kwargs):
        x, h = inputs
        y = self.h_transform_layer(h)
        x_filter = self.x_filter_layer(x)
        y_filter = self.y_filter_layer(y)
        x_gate = self.x_gate_layer(x)
        y_gate = self.y_gate_layer(y)
        z = tf.tanh(x_filter + y_filter) * tf.sigmoid(x_gate + y_gate)
        residual = self.res_layer(z)
        skip = self.skip_layer(z)
        return x + residual, skip


class MultiLayerPerceptronLayer(layers.Layer):
    """Multi-layer perceptron.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    n_units : int
        Number of units/neurons per layer.
    norm_type : str
        Either 'layer', 'batch' or None.
    act_type : 'str'
        Activation type, e.g. 'relu', 'elu', etc.
    drop_rate : float
        Dropout rate for the output of each layer.
    """

    def __init__(
        self,
        n_layers: int,
        n_units: int,
        norm_type: str,
        act_type: str,
        drop_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = []
        for n in range(n_layers):
            self.layers.append(layers.Dense(n_units))
            self.layers.append(NormalizationLayer(norm_type))
            self.layers.append(layers.Activation(act_type))
            self.layers.append(layers.Dropout(drop_rate))

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs


class CategoricalKLDivergenceLayer(layers.Layer):
    """Layer to calculate a KL divergence between two categorical distributions.

    Parameters
    ----------
    clip_start : int
        Index to clip the sequences inputted to this layer.
    """

    def __init__(self, clip_start: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.clip_start = clip_start

    def call(self, inputs, **kwargs):
        inference_logits, model_logits = inputs

        # The model network predicts one time step into the future compared to
        # the inference network. We clip the sequences to ensure we are comparing
        # the correct time points.
        model_logits = model_logits[:, self.clip_start : -1]
        inference_logits = inference_logits[:, self.clip_start + 1 :]

        # Calculate the KL divergence between the posterior and prior
        prior = tfp.distributions.Categorical(logits=model_logits)
        posterior = tfp.distributions.Categorical(logits=inference_logits)
        kl_loss = tfp.distributions.kl_divergence(
            posterior, prior, allow_nan_stats=False
        )

        # Sum the KL loss for each time point and average over batches
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = tf.reduce_mean(kl_loss, axis=0)

        return kl_loss


class CategoricalLogLikelihoodLossLayer(layers.Layer):
    """Layer to calculate the log-likelihood loss assuming a categorical model.

    Parameters
    ----------
    n_states : int
        Number of states
    """

    def __init__(self, n_states, **kwargs):
        super().__init__(**kwargs)
        self.n_states = n_states

    def call(self, inputs, **kwargs):
        x, mu, sigma, probs = inputs

        # Log-likelihood for each state
        ll_loss = tf.zeros(shape=tf.shape(x)[:-1])
        for i in range(self.n_states):
            mvn = tfp.distributions.MultivariateNormalTriL(
                loc=mu[i],
                scale_tril=tf.linalg.cholesky(sigma[i]),
                allow_nan_stats=False,
            )
            ll_loss += probs[:, :, i] * mvn.log_prob(x)

        # Sum over time dimension and average over the batch dimension
        ll_loss = tf.reduce_sum(ll_loss, axis=1)
        ll_loss = tf.reduce_mean(ll_loss, axis=0)

        # Add the negative log-likelihood to the loss
        nll_loss = -ll_loss
        self.add_loss(nll_loss)
        self.add_metric(nll_loss, name=self.name)

        return nll_loss


class ScalarLayer(layers.Layer):
    """Layer to learn a single scalar.

    Parameters
    ----------
    learn : bool
        Should we learn the scalar?
    initial_value : float
        Initial value for the scalar.
    """

    def __init__(
        self,
        learn: bool,
        initial_value: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.learn = learn
        if initial_value is None:
            self.initial_value = 1
        else:
            self.initial_value = initial_value.astype("float32")
        self.scalar_initializer = WeightInitializer(self.initial_value)

        def build(self):
            self.scalar = self.add_weight(
                "scalar",
                dtype=tf.float32,
                initializer=self.scalar_initializer,
                trainable=self.learn,
            )
            self.built = True

        def call(self, inputs, **kwargs):
            return self.scalar


class SubjectMeansCovsLayer(layers.Layer):
    """Class for subject specific means and covariances.

    Here:
    - mu_j^(s_t) = mu_j + delta_mu_j
    - D_j^(s_t)  = D_j  + delta_D_j

    where mu_j, D_j are group parameters, delta_mu_j and delta_D_j
    are subject specific deviations from the group parameters

    Parameters
    ----------
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    n_subjects : int
        Number of subjects.
    """

    def __init__(self, n_modes, n_channels, n_subjects, **kwargs):
        super().__init__(**kwargs)
        self.n_modes = n_modes
        self.n_channels = n_channels
        self.n_subjects = n_subjects

        # Bijector used to transform vectors to covariance matrices
        self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])
        self.delta_mu_subject_layer = layers.Dense(self.n_channels)
        self.flattened_delta_D_cholesky_factors_subject_layer = layers.Dense(
            self.n_channels * (self.n_channels + 1) // 2
        )
        self.delta_mu_mode_decoder_layer = layers.Dense(3)
        self.flattened_delta_D_cholesky_factors_mode_decoder_layer = layers.Dense(3)
        self.delta_mu_mode_layer = layers.Dense(self.n_channels)
        self.flattened_delta_D_cholesky_factors_mode_layer = layers.Dense(
            self.n_channels * (self.n_channels + 1) // 2
        )
        # attribute for initialising weights of the layers
        self.layers = [
            self.delta_mu_subject_layer,
            self.flattened_delta_D_cholesky_factors_subject_layer,
            self.delta_mu_mode_layer,
            self.flattened_delta_D_cholesky_factors_mode_layer,
            self.delta_mu_mode_decoder_layer,
            self.flattened_delta_D_cholesky_factors_mode_decoder_layer,
        ]

    def call(self, inputs):
        # group_mu.shape = (n_modes, n_channels)
        # group_D.shape = (n_modes, n_channels, n_channels)
        # subject_embeddings.shape = (n_subjects, embedding_dim)
        group_mu, group_D, subject_embeddings = inputs

        # This has shape (n_modes, n_channels * (n_channels + 1) // 2)
        flattened_group_D_cholesky_factors = self.bijector.inverse(group_D)

        # Compute the deviation of each subject from the group
        delta_mu_subject = self.delta_mu_subject_layer(subject_embeddings)
        flattened_delta_D_cholesky_factors_subject = (
            self.flattened_delta_D_cholesky_factors_subject_layer(subject_embeddings)
        )

        decode_delta_mu_mode = self.delta_mu_mode_decoder_layer(group_mu)
        delta_mu_mode = self.delta_mu_mode_layer(decode_delta_mu_mode)

        decode_flattened_delta_D_cholesky_factors_mode = (
            self.flattened_delta_D_cholesky_factors_mode_decoder_layer(
                flattened_group_D_cholesky_factors
            )
        )
        flattened_delta_D_cholesky_factors_mode = (
            self.flattened_delta_D_cholesky_factors_mode_layer(
                decode_flattened_delta_D_cholesky_factors_mode
            )
        )

        # Shapes:
        # delta_mu_subject.shape = (n_subjects, n_channels)
        # flattened_delta_D_cholesky_factors_subject.shape = (n_subjects, n_channels * (n_channels + 1) // 2)
        # delta_mu_mode.shape = (n_modes, n_channels)
        # flattened_delta_D_cholesky_factors_mode.shape = (n_modes, n_channels * (n_channels + 1) // 2)

        # Match the dimensions for addition
        group_mu = tf.expand_dims(group_mu, axis=0)
        delta_mu_mode = tf.expand_dims(delta_mu_mode, axis=0)
        delta_mu_subject = tf.expand_dims(delta_mu_subject, axis=1)
        mu = tf.add(tf.add(group_mu, delta_mu_mode), delta_mu_subject)

        flattened_group_D_cholesky_factors = tf.expand_dims(
            flattened_group_D_cholesky_factors, axis=0
        )
        flattened_delta_D_cholesky_factors_mode = tf.expand_dims(
            flattened_delta_D_cholesky_factors_mode, axis=0
        )
        flattened_delta_D_cholesky_factors_subject = tf.expand_dims(
            flattened_delta_D_cholesky_factors_subject, axis=1
        )
        flattened_D_cholesky_factors = tf.add(
            tf.add(
                flattened_group_D_cholesky_factors,
                flattened_delta_D_cholesky_factors_mode,
            ),
            flattened_delta_D_cholesky_factors_subject,
        )
        D = self.bijector(flattened_D_cholesky_factors)

        return mu, D


class MixSubjectEmbeddingParametersLayer(layers.Layer):
    """Class for mixing means and covariances for the
    subject embedding model.

    The mixture is calculated as
    - m_t = Sum_j alpha_jt mu_j^(s_t)
    - C_t = Sum_j alpha_jt D_j^(s_t)
    where s_t is the subject at time t.
    """

    def call(self, inputs):
        # alpha.shape = (None, sequence_length, n_modes)
        # mu.shape = (n_subjects, n_modes, n_channels)
        # D.shape = (n_subjects, n_modes, n_channels, n_channels)
        # subj_id.shape = (None, sequence_length)
        alpha, mu, D, subj_id = inputs
        subj_id = tf.cast(subj_id, tf.int32)

        # The parameters for each time point
        dynamic_mu = tf.gather(mu, subj_id)
        dynamic_D = tf.gather(D, subj_id)

        # Next mix with the time course
        alpha = tf.expand_dims(alpha, axis=-1)
        m = tf.reduce_sum(tf.multiply(alpha, dynamic_mu), axis=2)

        alpha = tf.expand_dims(alpha, axis=-1)
        C = tf.reduce_sum(tf.multiply(alpha, dynamic_D), axis=2)

        return m, C
