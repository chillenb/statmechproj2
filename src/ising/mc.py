from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._core import Model
from .makeplot import make_plot


class ModelWrap:
    def __init__(self, J = 1.0, hmu = 0.0, K = 20, seed=0):
        self.hmu = hmu
        self.J = J
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.choice([-1, 1], size=(K,K))
        self.model = Model(self.rng.bit_generator.random_raw(), J, hmu, np.zeros(400, dtype=int))#.flatten())

#     vector<Real> random_mc(vector<int>& x_rand, vector<int>& y_rand, vector<Real>& samp_rand, Real beta, int samp_freq) {

    def random_mc(self, beta, nsamp, samp_freq=100, verbose=True):
        x_rand = self.rng.integers(0, self.K, size=nsamp)
        y_rand = self.rng.integers(0, self.K, size=nsamp)
        samp_rand = self.rng.random(nsamp)
        energies = self.model.random_mc(x_rand, y_rand, samp_rand, beta, samp_freq)
        energies = np.asarray(energies)
        if verbose:
            avgE = np.mean(energies)
            stdev = np.std(energies)
            print(f"Average energy: {avgE:.4f} +/- {stdev:.4f}")
        return energies

    def random_mc_meanstd(self, beta, nsamp, samp_freq=100, verbose=True):
        mean, std = self.model.random_mc_meanstd(nsamp, beta, samp_freq)
        if verbose:
            print(f"Average energy: {mean:.4f} +/- {std:.4f}")
        return mean, std
    
    def random_mc_large(self, beta, nsamp, samp_freq=100, verbose=True, chunksize=1000000):
        c = min(chunksize, nsamp)
        nc = int(np.ceil(nsamp/c))
        means = np.zeros(nc)
        mags = np.zeros(nc)

        for i in range(nc):
            x_rand = self.rng.integers(0, self.K, size=c)
            y_rand = self.rng.integers(0, self.K, size=c)
            samp_rand = self.rng.random(c)
            mean, std, mag = self.model.random_mc_meanstd(nsamp, beta, samp_freq)
            means[i] = mean
            mags[i] = mag
            del x_rand, y_rand, samp_rand
        mm = np.mean(means)
        mmag = np.mean(mags)
        if verbose:
            print(f"Average energy: {mm:.4f}")
        return mm, mmag