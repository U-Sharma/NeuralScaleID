# Adapted from ID measurement code https://github.com/ansuini/IntrinsicDimDeep
# and extended for k-NNs, MLE, and the use of sklearn for Nearest Neighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model


def get_ratios(vectors, n_neighbors):
    N = len(vectors)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)
    ratios = np.array(
        [distances[:, i] / distances[:, 1] for i in range(2, n_neighbors)]
    ).T
    return ratios, N


def measure_dimension_kNN(
    vectors, n_neighbors=5, fraction=0.9, plot=False, verbose=False
):
    ratios, N = get_ratios(vectors, n_neighbors)
    mus = [
        np.sort(ratios[i], axis=None, kind="quicksort") for i in range(n_neighbors - 2)
    ]
    Femp = (np.arange(1, N + 1, dtype=np.float64)) / N

    dims = []
    xs = []
    ys = []
    for k, mu in enumerate(mus):
        x = np.log(mu[:-2])
        xs += [x]
        y = -np.log(1 - Femp[:-2] ** (1 / (k + 1)))
        ys += [y]

        # regression
        npoints = int(np.floor(N * fraction))
        regr = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(x[:npoints, np.newaxis], y[:npoints, np.newaxis])
        if verbose:
            print(
                "From ratio " + str(k + 2) + " NN estimated dim " + str(regr.coef_[0])
            )
        dims += [regr.coef_[0]]

    if plot:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("Log F vs Log mu")
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        for x, y in zip(xs, ys):
            ax.scatter(x, y)
            ax.plot(
                x[:npoints], regr.predict(x[:npoints].reshape((-1, 1))), color="gray"
            )
        plt.show()
    return xs, ys, dims


def measure_dimension_MLE(vectors, n_neighbors=10, plot=False, verbose=False):
    ratios, _ = get_ratios(vectors, n_neighbors)
    logs = np.log(ratios)
    estimates = (n_neighbors - 2) / (logs[:, -1] - np.sum(logs[:, :-1], axis=1))
    dim = np.mean(estimates)
    var = np.var(estimates)
    if verbose:
        print("Dimension MLE: ", dim, " Stddev: ", np.sqrt(var))
    if plot:
        fig, axs = plt.subplots(1, 1)
        axs.hist(estimates, bins=50)

    return dim, var, estimates
