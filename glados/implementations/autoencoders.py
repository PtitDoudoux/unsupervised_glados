#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Multiple implementations of the AutoEncoders algorithm
"""


from typing import Callable, List, Tuple, Union

from keras.layers import Conv2D, Dense, Input, Layer, MaxPooling2D, UpSampling2D
from keras import Model
import numpy as np


def _dense_architecture_builder(input_shape: Tuple, output_shape: int, neurons: List[int],
                                layer_activation='relu', output_activation='sigmoid') -> List[Layer]:
    """
    Build the keras architecture
    :param input_shape: The input shape of the model
    :param output_shape: The output shape of the model
    :param neurons: The number of neurons per layers
    :param layer_activation: The activation function to use within the dense layer
    :param output_activation: The activation function to use with the output layer
    :return: A list of Keras layer to to build a Keras Model
    """
    layers = [Input(input_shape)]
    for n in neurons:
        layers.append(Dense(n, activation=layer_activation))
    layers.append(Dense(output_shape, activation=output_activation))
    return layers


def _cnn_architecture_builder(input_shape: Tuple, neurons: List[int], conv=(3, 3), padding='same',
                              layer_activation='relu', output_activation='sigmoid', decoder=False)\
        -> List[Layer]:
    """
    Build the keras architecture
    :param input_shape: The input shape of the model
    :param neurons: The number of neurons per layers
    :param conv: The tuple representing the convolution to be applied
    :param padding: The padding to be applid on the convolution layers
    :param layer_activation: The activation function to use within the dense layer
    :param output_activation: The activation function to use with the output layer
    :param decoder: If the generated architecture is a decoder
    :return: A list of Keras layer to to build a Keras Model
    """
    layers = [Input(input_shape)]
    for n in neurons:
        layers.append(Conv2D(n, conv, activation=layer_activation, padding=padding))
        layers.append(UpSampling2D((2, 2)) if decoder else MaxPooling2D((2, 2), padding=padding))
    if decoder:
        layers.append(Conv2D(1, conv, padding=padding, activation=output_activation))
    return layers


def _model_builder(layers: List[Layer]) -> Model:
    """
    Build a Keras models based on a layer architecture
    :param layers: The layers to build the model
    :return: A builded (not compiled) Keras Model
    """
    inputs = layers[0]
    layer = layers[1](inputs)
    for l in layers[2:]:
        layer = l(layer)
    model = Model(inputs, layer)
    return model


def dense_encoder_decoder(input_shape: Tuple, output_shape: int, encoder_size: int, neurons: List[int],
                          optimizer: Union[Callable, str] = 'adam', loss: Union[Callable, str] = 'logcosh',
                          layers_activations=('relu', 'relu'), outputs_activations=('sigmoid', 'relu'))\
        -> Tuple[Model, Model, Model]:
    """
    Create the dense Encoder, Decoder and EncoderDecoder model
    :param input_shape: The input shape of the data
    :param output_shape: The output shape of the data
    :param encoder_size: The size of the encoder layer
    :param neurons: The number of neurons per layers
    :param optimizer: The optimizer to use
    :param loss: The loss function to use
    :param layers_activations: The tuple for the layers activation to use (encoder, decoder)
    :param outputs_activations: The tuple for the outputs activation to use (encoder, decoder)
    :return: An EncoderDecoder, compiled, Keras Model
    """
    encoder_layers = _dense_architecture_builder(input_shape, encoder_size, neurons,
                                                 layer_activation=layers_activations[0],
                                                 output_activation=outputs_activations[0])
    decoder_layers = _dense_architecture_builder((encoder_size,), output_shape, neurons[::-1],
                                                 layer_activation=layers_activations[1],
                                                 output_activation=outputs_activations[1])
    encoder_decoder_layers = encoder_layers + decoder_layers[1:]
    encoder = _model_builder(encoder_layers)
    decoder = _model_builder(decoder_layers)
    encoder_decoder = _model_builder(encoder_decoder_layers)
    encoder_decoder.compile(optimizer=optimizer, loss=loss)
    return encoder_decoder, encoder, decoder


def cnn_encoder_decoder(input_shape: Tuple, neurons: List[int],
                        optimizer: Union[Callable, str] = 'adam', loss: Union[Callable, str] = 'logcosh',
                        layers_activation='relu', outputs_activation='sigmoid')\
        -> Tuple[Model, Model, Model]:
    """
    Create the CNN Encoder, Decoder and EncoderDecoder model
    :param input_shape: The input shape of the data
    :param neurons: The number of neurons per layers
    :param optimizer: The optimizer to use
    :param loss: The loss function to use
    :param layers_activation: The tuple for the layers activation to use (encoder, decoder)
    :param outputs_activation: The tuple for the outputs activation to use (encoder, decoder)
    :return: An EncoderDecoder, compiled, Keras Model
    """
    encoder_layers = _cnn_architecture_builder(input_shape, neurons, layer_activation=layers_activation)
    encoder = _model_builder(encoder_layers)
    decoder_layers = _cnn_architecture_builder(input_shape, neurons[::-1], decoder=True,
                                               output_activation=outputs_activation,
                                               layer_activation=layers_activation)
    decoder = _model_builder(decoder_layers)
    encoder_decoder_layers = encoder_layers + decoder_layers[1:]
    encoder_decoder = _model_builder(encoder_decoder_layers)
    encoder_decoder.compile(optimizer=optimizer, loss=loss)
    return encoder_decoder, encoder, decoder


def generate_elements(decoder: Model, data: np.ndarray) -> np.ndarray:
    """
    Generate multiple random image from a decoder
    :param decoder: The decoder to use to generate the images
    :param data: The data to use to generate the images
    :return: A numpy array containing all the generated images
    """
    return decoder.predict(data, verbose=0)
