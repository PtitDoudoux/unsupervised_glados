#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Multiple implementations of the GAN algorithm
"""


from random import randint
from typing import List

from keras import Model
from keras.layers import Dense, Input
import numpy as np


class VanillaGAN:

    def __init__(self, data: np.ndarray, latent_space: int, generator: Model = None,
                 discriminator: Model = None):
        """
        Initialize a Vanilla GAN object
        :param data: The real data to use in the training process
        :param latent_space: The size of the entry latent space
        :param generator: Keras model generator to use if any
        :param discriminator: Keras model discriminator to use if any
        """
        self.data = data
        self.latent_space = latent_space
        self.generator = generator
        self.discriminator = discriminator

    def build(self, which: str, layers: List[int], layers_activation='relu',
              output_activation='relu', optimizer='adam', loss='logcosh') -> None:
        """
        Build a generator or discriminator model and store it inside the VanillaGAN object
        :param which: The type of model to generate (generator or discriminator)
        :param layers: The layers to use (size of each)
        :param layers_activation: The activation function to use within the layers
        :param output_activation: The activation function to use in the output layer
        :param optimizer: The optimizer function to use
        :param loss: The loss function to use
        """
        inputs = Input(self.data.shape[1:])
        layer = Dense(layers[1], activation=layers_activation)(inputs)
        for l in layers[2:-1]:
            layer = Dense(l, activation=layers_activation)(layer)
        layer = Dense(layers[-1], activation=output_activation)(layer)
        model = Model(inputs, layer)
        model.compile(optimizer=optimizer, loss=loss)
        if which == 'generator':
            self.generator = model
        else:
            self.discriminator = model

    def _discriminator_learning(self, batch_size=50, iteration=1, split=0.8, **kwargs) -> None:
        """
        Make the discriminator learn
        :param batch_size: The size of the batch to train on
        :param iteration: The number of iteration to do in the learning process of the discriminator
        :param split: The ratio in which to split the data
        :param kwargs: The keywords arguments to pass to the keras Model.fit method
        """
        discriminator_data = self._generate_discriminator_batch(batch_size)
        x_train = [dd[0] for dd in discriminator_data]
        y_train = [dd[1] for dd in discriminator_data]
        self.discriminator.fit(x_train, y_train, epochs=iteration, validation_split=split, **kwargs)

    def _generate_discriminator_batch(self, batch_size: int) -> np.ndarray:
        """
        Generate a batch for the GAN discriminator model with fake and real data shuffled
        :param batch_size: The size of the batch to generate
        :return: A batch shuffled with fake and real data
        """
        data_size = self.data.shape[0]
        random_real_data = [[self.data[randint(0, data_size)], 1] for _ in range(int(batch_size/2))]
        fake_data = [[fd, 0] for fd in self.generator.predict(int(batch_size/2), verbose=0)]
        discriminator_data = np.concatenate((random_real_data, fake_data), axis=0)
        # np.random.shuffle(discriminator_data)
        return discriminator_data

    def fit(self, *args, **kwargs) -> None:
        """
        Fit the DenseEncoderDecoder model to the passed data
        :param args: The args to pass to keras model fit method
        :param kwargs: The keyword args to pass to keras model fit method
        """
        # self.model.fit(*args, *kwargs)
        raise NotImplementedError

    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError
