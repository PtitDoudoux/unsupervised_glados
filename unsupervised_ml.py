#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Test implementations
"""


from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from glados.implementations import CNNAutoEncoder, DenseAutoEncoder, LloydKmeans, PCA
from glados.implementations.autoencoders import generate_elements as autoencoders_generate_elements


(x_train, _), (x_test, _) = mnist.load_data()


def run_dense_autoencoders():
    xtrain_ae = np.reshape(x_train.astype('float32'), (-1, 784)) / 255.0
    xtest_ae = np.reshape(x_test.astype('float32'), (-1, 784)) / 255.0
    dense_auto_encoder = DenseAutoEncoder((784,), 784, 2, [256, 128, 64, 32])
    dense_auto_encoder.build()
    dense_auto_encoder.encoder_decoder.fit(xtrain_ae, xtrain_ae, batch_size=256, epochs=25,
                                           shuffle=True, validation_data=(xtest_ae, xtest_ae))
    generated_inputs = np.asarray([[x, y] for x in np.arange(0, 1, 0.1)
                                   for y in np.arange(0, 1, 0.1)], dtype=np.float32)
    generated_imgs = autoencoders_generate_elements(dense_auto_encoder.decoder, generated_inputs)
    for img in generated_imgs[0::10]:
        Image.fromarray(np.reshape(img * 255.0, (28, 28))).show()


def run_cnn_autoencoders():
    xtrain_ae = np.reshape(x_train.astype('float32'), (-1, 28, 28, 1)) / 255.0
    xtest_ae = np.reshape(x_test.astype('float32'), (-1, 28, 28, 1)) / 255.0
    cnn_auto_encoder = CNNAutoEncoder((28, 28, 1), None, 2, [128, 64, 32])
    cnn_auto_encoder.build()
    cnn_auto_encoder.encoder_decoder.fit(xtrain_ae, xtrain_ae, batch_size=256, epochs=5,
                                         shuffle=True, validation_data=(xtest_ae, xtest_ae))
    generated_inputs = np.asarray([[x, y] for x in np.arange(0, 1, 0.1)
                                   for y in np.arange(0, 1, 0.1)], dtype=np.float32)
    generated_imgs = autoencoders_generate_elements(cnn_auto_encoder.decoder, generated_inputs)
    for img in generated_imgs[0::10]:
        Image.fromarray(np.reshape(img * 255.0, (28, 28))).show()


def run_kmeans():
    lloyd_kmeans = LloydKmeans(np.reshape(x_train, (-1, 784)), 10)
    lloyd_kmeans.fit()
    # count_by_mu = {mk: len(mu.cluster_item) for mk, mu in lloyd_kmeans.mus.items()}
    for mk, mu in lloyd_kmeans.mus.items():
        Image.fromarray(np.reshape(mu.representative, (28, 28))).show(mk)


def run_pca():
    pca_data = np.reshape(x_train, (-1, 784)) / 255.0
    pca = PCA(pca_data)
    plt.plot(pca.extracted_data[0], pca.extracted_data[1], 'ro')
    plt.show()


if __name__ == '__main__':
    # run_kmeans()
    # run_pca()
    # run_dense_autoencoders()
    run_cnn_autoencoders()
