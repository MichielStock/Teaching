"""
Created on Wednesday 18 October 2017
Last update: Thursday 19 October 2017

@author: Michiel Stock
michielfmstock@gmail.com

General class for optimal transport
"""

from sinkhorn_knopp import compute_optimal_transport
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import RidgeCV
import numpy as np

class OptimalTransport(object):
    """
    General class for optimal transport on a set of points with a given
    distance.
    """
    def __init__(self, X1, X2, M=None, r=None, c=None, lam=10,
                        fit_mapping=False, distance_metric='euclidean'):
        super(OptimalTransport, self).__init__()
        # check if all densities are nonzero
        assert (r is None or np.all(r>0)) and (c is None or np.all(c>0))
        self.X1 = X1
        n1, p1 = X1.shape
        self.X2 = X2
        n2, p2 = X2.shape
        self.lam = lam
        if r is None:  # uniform weights
            self.r = np.ones(n1) / n1
        else:
            self.r = r
        if c is None:  # uniform weights
            self.c = np.ones(n2) / n2
        else:
            self.c = c
        # compute the distance matrix
        if M is None:  # compute distance matrix
            self.M = pairwise_distances(X1, X2, metric=distance_metric)
        else:
            self.M = M
        # compute the optimal transport mapping
        self.compute_optimal_transport(lam, fit_mapping)
        if fit_mapping:
            self.fit_mapping()


    def compute_optimal_transport(self, lam, fit_mapping=False):
        """
        (Re)computes the optimal transport matrix using the Skinkhorn-Knopp
        algorithm

        Inputs:
            - lam : the value of the entropic regularization
            - fit_mapping : fit the mappings from and to the distributions (default=False)
        """
        self.P, self.d = compute_optimal_transport(
                                            self.M,self.r, self.c,
                                            lam, epsilon=1e-6)
        if fit_mapping:
            self.fit_mapping()

    def fit_mapping(self):
        """
        Fits the mappings from one distributions to the other
        """
        X1 = self.X1
        n1, p1 = X1.shape
        X2 = self.X2
        n2, p2 = X2.shape
        P = self.P
        c = self.c
        r = self.r
        reg_mapping = self.reg_mapping
        # mapping from X1 to X2
        self.model1to2 = RidgeCV(alphas=np.logspace(-3, 3, 7))
        self.model1to2.fit(X1, (P * c.reshape((-1, 1))) @ X2)
        # mapping from X2 to X1
        self.model2to1 = RidgeCV(alphas=np.logspace(-3, 3, 7))
        self.model2to1.fit(X2, (P.T * r.reshape((-1, 1))) @ X2)

    def interpolate(self, alpha):
        """
        Interpolate between the two distributions.

        Input:
            - alpha : value between 0 and 1 for the interpolation

        Output:
            - X : the interpolation between X1 and X2
            - w : weights of the points
        """
        mixing = self.P.copy()
        mixing /= self.r.reshape((-1, 1))
        X = (1 - alpha) * self.X1 + alpha * mixing @ self.X2
        w = (1 - alpha) * self.r + alpha * mixing @ self.c
        return X, w

    def mapX1toX2(self, X):
        """
        Map the first distribution to the second
        """
        return model1to2.predict(X)

    def mapX2toX1(self, X):
        """
        Map the second distribution to the first
        """
        return model2to1.predict(X)
