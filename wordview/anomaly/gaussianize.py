"""
Transform data so that it is approximately normally distributed

This code written by Greg Ver Steeg, 2015.
"""

import numpy as np
import sklearn
from scipy.optimize import fmin  # TODO: Explore efficacy of other opt. methods
from scipy.special import lambertw
from scipy.stats import boxcox, kurtosis, norm, rankdata

np.seterr(all="warn")


class Gaussianize(sklearn.base.TransformerMixin):
    """
    Gaussianize data using various methods.

    Conventions
    ----------
    This class is a wrapper that follows sklearn naming/style (e.g. fit(X) to train).
    In this code, x is the input, y is the output. But in the functions outside the class, I follow
    Georg's convention that Y is the input and X is the output (Gaussianized) data.

    Parameters
    ----------
    tol : float, default = 1e-4

    max_iter : int, default = 100
        Maximum number of iterations to search for correct parameters of Lambert transform.

    strategy : str, default='lambert'
        Possibilities are 'lambert'[1], 'brute'[2] and 'boxcox'[3].

    Attributes
    ----------
    coefs_ : list of tuples
        For each variable, we have transformation parameters.
        For Lambert, e.g., a tuple consisting of (mu, sigma, delta), corresponding to the parameters of the
        appropriate Lambert transform. Eq. 6 and 8 in the paper below.

    References
    ----------
    [1] Georg Goerg. The Lambert Way to Gaussianize heavy tailed data with
                        the inverse of Tukey's h transformation as a special case
        Author generously provides code in R: https://cran.r-project.org/web/packages/LambertW/
    [2] Valero Laparra, Gustavo Camps-Valls, and Jesus Malo. Iterative Gaussianization: From ICA to Random Rotations
    [3] Box cox transformation and references: https://en.wikipedia.org/wiki/Power_transform
    """

    def __init__(self, tol=1.22e-4, max_iter=100, verbose=False, strategy="lambert"):
        self.tol = tol
        self.max_iter = max_iter
        self.strategy = strategy
        self.coefs_ = []  # Store tau for each transformed variable
        self.verbose = verbose

    def fit(self, x, y=None):
        """Fit a Gaussianizing transformation to each variable/column in x."""
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        elif len(x.shape) != 2:
            print(
                "Data should be a 1-d list of samples to transform or a 2d array with samples as rows."
            )

        if self.strategy == "lambert":
            if self.verbose:
                print("Gaussianizing with Lambert method")
            for x_i in x.T:
                self.coefs_.append(igmm(x_i, tol=self.tol, max_iter=self.max_iter))
        elif self.strategy == "brute":
            for x_i in x.T:
                self.coefs_.append(
                    None
                )  # TODO: In principle, we could store parameters to do a quasi-invert
        elif self.strategy == "boxcox":
            for x_i in x.T:
                self.coefs_.append(boxcox(x_i)[1])
        else:
            raise NotImplementedError
        return self

    def transform(self, x):
        """Transform new data using a previously learned Gaussianization model."""
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        elif len(x.shape) != 2:
            print(
                "Data should be a 1-d list of samples to transform or a 2d array with samples as rows."
            )
        if x.shape[1] != len(self.coefs_):
            print(
                "%d variables in test data, but %d variables were in training data."
                % (x.shape[1], len(self.coefs_))
            )

        if self.strategy == "lambert":
            return np.array([w_t(x_i, tau_i) for x_i, tau_i in zip(x.T, self.coefs_)]).T
        elif self.strategy == "brute":
            return np.array(
                [norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]
            ).T
        elif self.strategy == "boxcox":
            return np.array(
                [boxcox(x_i, lmbda=lmbda_i) for x_i, lmbda_i in zip(x.T, self.coefs_)]
            ).T
        else:
            raise NotImplementedError

    def inverse_transform(self, y):
        """Recover original data from Gaussianized data."""
        if self.strategy == "lambert":
            return np.array(
                [inverse(y_i, tau_i) for y_i, tau_i in zip(y.T, self.coefs_)]
            ).T
        elif self.strategy == "boxcox":
            return np.array(
                [
                    (1.0 + lmbda_i * y_i) ** (1.0 / lmbda_i)
                    for y_i, lmbda_i in zip(y.T, self.coefs_)
                ]
            ).T
        else:
            print("Inversion not supported for this gaussianization transform.")
            raise NotImplementedError

    def qqplot(self, x, prefix="qq"):
        """Show qq plots compared to normal before and after the transform."""
        from matplotlib import pylab
        from scipy.stats import probplot

        y = self.transform(x)

        for i, (x_i, y_i) in enumerate(zip(x.T, y.T)):
            probplot(x_i, dist="norm", plot=pylab)
            pylab.savefig(prefix + "_%d_before.png" % i)
            pylab.clf()

            probplot(y_i, dist="norm", plot=pylab)
            pylab.savefig(prefix + "_%d_after.png" % i)
            pylab.clf()


def w_d(z, delta):
    # Eq. 9
    if delta < 1e-6:
        return z
    return np.sign(z) * np.sqrt(np.real(lambertw(delta * z**2)) / delta)


def w_t(y, tau):
    # Eq. 8
    return tau[0] + tau[1] * w_d((y - tau[0]) / tau[1], tau[2])


def inverse(x, tau):
    # Eq. 6
    u = (x - tau[0]) / tau[1]
    return tau[0] + tau[1] * (u * np.exp(u * u * (tau[2] * 0.5)))


def igmm(y, tol=1.22e-4, max_iter=100):
    # Infer mu, sigma, delta using IGMM in Alg.2, Appendix C
    if np.std(y) < 1e-4:
        return np.mean(y), np.std(y).clip(1e-4), 0
    delta0 = delta_init(y)
    tau1 = (np.median(y), np.std(y) * (1.0 - 2.0 * delta0) ** 0.75, delta0)
    for k in range(max_iter):
        tau0 = tau1
        z = (y - tau1[0]) / tau1[1]
        delta1 = delta_gmm(z)
        x = tau0[0] + tau1[1] * w_d(z, delta1)
        mu1, sigma1 = np.mean(x), np.std(x)
        tau1 = (mu1, sigma1, delta1)

        if np.linalg.norm(np.array(tau1) - np.array(tau0)) < tol:
            break
        else:
            if k == max_iter - 1:
                print(
                    "Warning: No convergence after %d iterations. Increase max_iter."
                    % max_iter
                )
    return tau1


def delta_gmm(z):
    # Alg. 1, Appendix C
    delta0 = delta_init(z)

    def func(q):
        u = w_d(z, np.exp(q))
        if not np.all(np.isfinite(u)):
            return 0.0
        else:
            k = kurtosis(u, fisher=True, bias=False) ** 2
            if not np.isfinite(k) or k > 1e10:
                return 1e10
            else:
                return k

    res = fmin(func, np.log(delta0), disp=0)
    return np.around(np.exp(res[-1]), 6)


def delta_init(z):
    gamma = kurtosis(z, fisher=False, bias=False)
    with np.errstate(all="ignore"):
        delta0 = np.clip(1.0 / 66 * (np.sqrt(66 * gamma - 162.0) - 6.0), 0.01, 0.48)
    if not np.isfinite(delta0):
        delta0 = 0.01
    return delta0
