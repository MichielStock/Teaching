"""
Created on Friday 6 October 2017
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of the Sinkhorn-Knopp algorithm for optimal transport
"""

import numpy as np

def compute_optimal_transport(M, r, c, lam, epsilon=1e-5):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(- lam * M)
    P /= P.sum()
    u = np.zeros(n)
    # normalize this matrix
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, np.sum(P * M)


if __name__ == '__main__':

    from sys import path
    path.append('../../Scripts')
    from plotting import plt, blue, orange, green, red, yellow


    from sklearn.datasets import make_moons
    from scipy.spatial import distance_matrix

    X, y = make_moons(n_samples=100, noise=0.1, shuffle=False)

    X1 = X[y==1,:]
    X2 = -X[y==0,:]

    X2 = X2[:40]  # different size X1 and X2

    n, m = X1.shape[0], X2.shape[0]

    r = np.ones(n) / n
    c = np.ones(m) / m

    M = distance_matrix(X1, X2)

    P, d = compute_optimal_transport(M, r, c, lam=30, epsilon=1e-5)

    fig, (ax1, ax2, ax) = plt.subplots(ncols=3)
    ax.scatter(X1[:,0], X1[:,1], color=blue)
    ax.scatter(X2[:,0], X2[:,1], color=orange)

    for i in range(n):
        for j in range(m):
            ax.plot([X1[i,0], X2[j,0]], [X1[i,1], X2[j,1]], color=red,
                    alpha=P[i,j] * n)

    ax.set_title('Optimal matching')

    ax1.imshow(M)
    ax1.set_title('Cost matrix')

    ax2.imshow(P)
    ax2.set_title('Transport matrix')
    plt.show()
