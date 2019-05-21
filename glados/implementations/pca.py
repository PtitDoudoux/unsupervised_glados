#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Multiple implementations of the PCA algorithm
"""


import numpy as np


def _calculate_covariance_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Determinate the covariance matrix of two vector
    :param v1: The first vector
    :param v2: The second vector
    :return: The resultant covariance matrix
    """
    return (1 / (len(v1) - 1)) * sum(((v1[i] - np.mean(v1)) * (v2[i] - np.mean(v2)) for i in range(len(v1))))


covariance_matrix = np.vectorize(lambda d: np.asarray([[_calculate_covariance_matrix(x, y) for x in d] for y in d]))


def principal_component_analysis(data: np.ndarray, components=2) -> np.ndarray:
    """
    Apply the PCA algorithm onto a Matrix
    :param data: The data matrix to apply PCA on
    :param components: The number of components to extract from the projected matrix
    :return: The projected matrix in n dimensions
    """
    centered_data = data - np.mean(data.T, axis=1)
    cov_matrix = np.asarray([[_calculate_covariance_matrix(x, y) for x in centered_data] for y in centered_data])
    eigen_val, eigen_vec = np.linalg.eigh(cov_matrix)
    order = (-eigen_val).argsort()
    eigen_vec_sorted = np.transpose(eigen_vec)[order]
    return eigen_vec_sorted[0:components].dot(centered_data)
