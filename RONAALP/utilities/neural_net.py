#####################################################################
# %% IMPORTING MODULES

import keras
from keras import layers

#####################################################################


def create_autoencoder(architecture, input_dim, latent_dim, output_dim=None):
    """
    Create a flexible autoencoder model with a symmetrical architecture for encoding and decoding.

    Parameters
    ----------
    architecture : list
            A list specifying the number of neurons in each hidden layer for the encoding and decoding parts.
    input_dim : int
            The dimension of the input data.
    latent_dim : int
            The dimension of the bottleneck layer (latent space).
    output_dim : int, optional
            The dimension of the output data. If None, it defaults to the input_dim.

    Returns
    -------
    autoencoder : keras.models.Model
            The autoencoder model with both encoding and decoding parts.
    encoder : keras.models.Model
            The encoder model only.
    """
    if output_dim is None:
        output_dim = input_dim

    encoding_layers = []
    decoding_layers = []
    input_layer = keras.Input(shape=(input_dim,))

    # Encoding layers
    for num_neurons in architecture:
        encoding_layer = layers.Dense(num_neurons, activation="tanh")(
            input_layer if not encoding_layers else encoding_layers[-1]
        )
        encoding_layers.append(encoding_layer)

    encoded = layers.Dense(latent_dim, activation="linear")(encoding_layers[-1])

    # Decoding layers (symmetrical to encoding layers)
    for num_neurons in reversed(architecture):
        decoding_layer = layers.Dense(num_neurons, activation="tanh")(
            encoded if not decoding_layers else decoding_layers[-1]
        )
        decoding_layers.append(decoding_layer)

    decoded = layers.Dense(output_dim, activation="linear")(decoding_layers[-1])

    autoencoder = keras.Model(input_layer, decoded)
    encoder = keras.Model(input_layer, encoded)

    return autoencoder, encoder
