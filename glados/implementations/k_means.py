#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Multiple implementations of the K-Means algorithm
"""


from operator import itemgetter
from random import randint
from typing import Any, Dict, List, NamedTuple

import numpy as np
from tqdm import tqdm


class MU(NamedTuple):
    representative: Any
    cluster_item: List[Any]


def lloyd_kmeans(data: np.ndarray, nb_clusters: int, max_iteration=100) -> Dict:
    """
    Lloyd's algorithm for the K-Means
    :param data: The Array to compute the K-Means on
    :param nb_clusters: The number of cluster find
    :param max_iteration: The maximum number of iteration before stopping the algorithm
    :return: A dict with the cluster id and the list of point associated
    """
    mus = {f'mu{nc}': MU(data[randint(0, data.shape[0])], list()) for nc in range(nb_clusters)}
    iteration = 1
    while "we haven't find each mu":
        print(f'Iteration {iteration}')
        pbar = tqdm(total=100)
        for item in data:
            mu_distance = ((mu_id, np.linalg.norm(item - mu.representative)) for mu_id, mu in mus.items())
            min_mu_id = sorted(mu_distance, key=itemgetter(1))[0][0]
            mus[min_mu_id].cluster_item.append(item)
            pbar.update((1 / len(data)) * 100)
        pbar.close()
        new_mus = {mu_id: MU(np.mean(mu.cluster_item, axis=0), list()) for mu_id, mu in mus.items()}
        is_finished = all(((mus[mu_id].representative == new_mus[mu_id].representative).all() for mu_id in mus.keys()))
        if is_finished or iteration > max_iteration:
            break
        mus = new_mus
        iteration += 1
    return mus


def generate_elements(mu1: np.ndarray, mu2: np.ndarray, gradient: float) -> np.ndarray:
    """
    Generate a random element
    :param mu1: The first mu to base the generation on
    :param mu2: The second mu to base the generation on
    :param gradient: The gradient between two centroid [0:1]
    :return: The generated element
    """
    return mu1 + ((mu1 - mu2) * gradient)
