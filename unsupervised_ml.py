#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Test implementations
"""


from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from glados.implementations.k_means import lloyd_kmeans
from glados.implementations import autoencoders, pca


(x_train, _), (x_test, _) = mnist.load_data()


def run_dense_autoencoders():
    xtrain_ae = np.reshape(x_train.astype('float32'), (-1, 784)) / 255.0
    xtest_ae = np.reshape(x_test.astype('float32'), (-1, 784)) / 255.0
    encoder_decoder, encoder, decoder = \
        autoencoders.dense_encoder_decoder((784,), 784, 2, [256, 128, 64, 32])
    encoder_decoder.fit(xtrain_ae, xtrain_ae, batch_size=256, epochs=25,
                        shuffle=True, validation_data=(xtest_ae, xtest_ae))
    generated_inputs = np.asarray([[x, y] for x in np.arange(0, 1, 0.1)
                                   for y in np.arange(0, 1, 0.1)], dtype=np.float32)
    generated_imgs = autoencoders.generate_elements(decoder, generated_inputs)
    for img in generated_imgs:
        Image.fromarray(np.reshape(img * 255.0, (28, 28))).show()


def run_cnn_autoencoders():
    xtrain_ae = np.reshape(x_train.astype('float32'), (-1, 28, 28, 1)) / 255.0
    xtest_ae = np.reshape(x_test.astype('float32'), (-1, 28, 28, 1)) / 255.0
    encoder_decoder, encoder, decoder = \
        autoencoders.cnn_encoder_decoder((28, 28, 1), [128, 64, 32])
    encoder_decoder.fit(xtrain_ae, xtrain_ae, batch_size=256, epochs=25,
                        shuffle=True, validation_data=(xtest_ae, xtest_ae))
    generated_inputs = np.asarray([[x, y] for x in np.arange(0, 1, 0.1)
                                   for y in np.arange(0, 1, 0.1)], dtype=np.float32)
    generated_imgs = autoencoders.generate_elements(decoder, generated_inputs)
    for img in generated_imgs:
        Image.fromarray(img * 255.0).show()


def run_kmeans():
    kmeans_mnist = lloyd_kmeans(np.reshape(x_train, (60000, 784)), 10)
    res = {mk: len(mu.cluster_item) for mk, mu in kmeans_mnist.items()}
    for mk, mu in kmeans_mnist.items():
        Image.fromarray(np.reshape(mu.representative, (28, 28))).show(mk)


def run_pca():
    pca_data = np.reshape(x_train[0:50], (50, 784)) / 255.0
    res = pca.principal_component_analysis(pca_data)
    plt.plot(res[0], res[1], 'ro')
    plt.show()


if __name__ == '__main__':
    # run_kmeans()
    # run_pca()
    run_dense_autoencoders()
    # run_cnn_autoencoders()
