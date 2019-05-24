#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Some runs with our Kmeans implementations on different data
"""


from keras.datasets import mnist
import numpy as np
# from PIL import Image

from glados.implementations import LloydKmeans
from glados.utils import plot_elements, load_sign_language_digits_ds


(x_train, _), (x_test, _) = mnist.load_data()
fake_data = np.asarray([[1, 3], [1.5, 2.5], [1, 2], [3, 1], [3, 2]])


# Kmeans on fake data
fake_lloyd_kmeans = LloydKmeans(fake_data / 3.0, 2)
fake_lloyd_kmeans.fit()
print({mk: np.asarray(mu.cluster_item)*3.0 for mk, mu in fake_lloyd_kmeans.mus.items()})


# Kmeans run on MNIST
mnist_lloyd_kmeans = LloydKmeans(np.reshape(x_train, (-1, 784)) / 255.0, 15)
mnist_lloyd_kmeans.fit()
mus_imgs = np.asarray([np.reshape(mu.representative, (28, 28)) * 255.0 for mu in mnist_lloyd_kmeans.mus.values()])
plot_elements(mus_imgs, 5, 3)
# generated_img = LloydKmeans.generate(mnist_lloyd_kmeans.mus['mu1'], mnist_lloyd_kmeans.mus['mu2'], 0.5)
# generated_img = np.reshape(generated_img, (28, 28)) * 255.0
# Image.fromarray(generated_img).show()


# Kmeans on hand Dataset
dhi = load_sign_language_digits_ds('./../data/Sign-Language-Digits-Dataset-master', (28, 28))
dhi_train = np.reshape(dhi.astype('float32'), (-1, 784)) / 255.0
dhi_lloyd_kmeans = LloydKmeans(dhi_train, 15)
dhi_lloyd_kmeans.fit()
mus_imgs = np.asarray([np.reshape(mu.representative, (28, 28)) * 255.0 for mu in dhi_lloyd_kmeans.mus.values()])
plot_elements(mus_imgs, 5, 3)
# generated_img = LloydKmeans.generate(dhi_lloyd_kmeans.mus['mu1'], dhi_lloyd_kmeans.mus['mu2'], 0.5)
# generated_img = np.reshape(generated_img, (28, 28)) * 255.0
