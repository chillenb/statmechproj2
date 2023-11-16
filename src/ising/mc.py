import numpy as np

from ._core import Model


class ModelWrap:
    def __init__(self, J = 1.0, hmu = 0.0, K = 20, seed=0):
        self.hmu = hmu
        self.J = J
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.choice([-1, 1], size=(K,K))
        self.model = Model(self.rng.bit_generator.random_raw(), J, hmu, np.zeros(400, dtype=int))#.flatten())

    def random_mc_meanstd(self, beta, nsamp, samp_freq=100, verbose=True):
        mean, std, mag = self.model.random_mc_meanstd(nsamp, beta, samp_freq)
        if verbose:
            print(f"<E>: {mean:.4f} <M>: {mag:.4f}")
        return mean, std, mag