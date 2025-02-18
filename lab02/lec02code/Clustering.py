#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt

# Import abstract
from abc import ABC, abstractmethod


class Clustering(ABC):
    """
    Abstract class for K-means-like clustering:
    Minimizes SSE from points to cluster centers.
    """

    # ------------------------------------------------------------
    # (1) Generate Synthetic Data (Optional)
    # ------------------------------------------------------------
    def _generate_gaussian_blobs(
        self, num_points_per_blob=60, centers=[(0, 0), (5, 5)], std=1.0
    ):
        """
        Generate synthetic 2D data from multiple Gaussian blobs.
        """
        all_points = []
        for cx, cy in centers:
            blob = np.random.normal(
                loc=(cx, cy), scale=std, size=(num_points_per_blob, 2)
            )
            all_points.append(blob)
        data = np.vstack(all_points)
        return data
