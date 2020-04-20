import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model


def measure_dimension_fast(vectors, n_neighbors=5, plot=False, fraction_late=0.0, fraction_early=0.9,verbose=1):
    N = len(vectors)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)
    ratios = [distances[:,i] / distances[:,1] for i in range(2, n_neighbors)]
    mus = [np.sort(ratios[i], axis=None, kind='quicksort') for i in range(n_neighbors-2)]
    Femp = (np.arange(1,N+1,dtype=np.float64) )/N
    #mus = [m[~m.isnan()] for m in mus]
    
    dims = []
    xs = []
    ys = []
    for k, mu in enumerate(mus):
        # take logs (leave out the last element because 1-Femp is zero there)
        x = np.log(mu[:-2])
        xs += [x]
        y = -np.log(1 - Femp[:-2]**(1 / (k + 1)))
        #y = y[:len(x)]
        ys += [y]

        # regression
        #start_points_near_end = int(np.floor(N*fraction_late))
        npoints = int(np.floor(N*fraction_early))
        regr = linear_model.LinearRegression(fit_intercept=True)
        #regr_end = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(x[:npoints,np.newaxis],y[:npoints,np.newaxis]) 
        #regr_end.fit(x[start_points_near_end:,np.newaxis],y[start_points_near_end:,np.newaxis]) 
        if verbose>0: print("From ratio " + str(k+2) + " NN estimated dim " + str(regr.coef_[0]))
        #print("NN estimated dim from END: " + str(regr_end.coef_[0]))
        #print("Score: " + str(regr.score(x[:npoints,np.newaxis],y[:npoints,np.newaxis])))
        dims += [regr.coef_[0]]

    if plot:
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_title("Log F vs Log mu")
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        #print(regr.predict(x[:npoints,np.newaxis]))
        for x, y in zip(xs, ys):
            ax.scatter(x, y)
            ax.plot(x[:npoints], regr.predict(x[:npoints].reshape((-1,1))), color='gray')
            #ax.plot(x[start_points_near_end:], regr_end.predict(x[start_points_near_end:].reshape((-1,1))), color='gray')
        plt.show()
    return xs, ys, dims


def MLE_estimate(ratios, n_neighbors):
    logs = np.log(ratios)
    estimates = logs[:,-1] - np.sum(logs[:, :-1], axis=1) / (n_neighbors - 1)
    return estimates

def measure_dimension_MLE(vectors, n_neighbors=100, plot=False):
    N = len(vectors)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)
    ratios = np.array([distances[:,i] / distances[:,1] for i in range(2, n_neighbors)]).T
    estimates = MLE_estimate(ratios, n_neighbors)
    dim = np.mean(1/estimates)
    var = np.var(1/estimates)
    print("Dimension MLE: ", dim, " Stddev: ", np.sqrt(var))
    axs = None
    if plot:
            fig, axs = plt.subplots(1, 1)
            axs.hist(1/estimates, bins=50)
            
    return dim, var, 1/estimates