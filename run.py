import multiprocessing as mp

import IPython
import matplotlib.pyplot as plt
import numpy as np
import psutil

import ising

temps = np.flip(np.hstack([np.linspace(0.01,4, 100)]))
betas = 1/temps

def warmup(mod, beta):
    mod.random_mc_meanstd(beta, 1000000, 1, verbose=False)

aves = np.zeros_like(betas)
stds = np.zeros_like(betas)

def measurebeta(b, seed):
    m = ising.ModelWrap(seed=seed)
    warmup(m, b)
    return m.random_mc_meanstd(b, 400000000, 1, verbose=True)

ncpus = psutil.cpu_count(logical=True)
if not ncpus:
    ncpus = mp.cpu_count()
with mp.Pool(ncpus) as pool:
    seq = np.random.SeedSequence(12345)
    ss = list(seq.spawn(len(betas)))
    data = pool.starmap(measurebeta, zip(list(betas), ss))

aves = np.array([d[0] for d in data])
mags = np.array([d[2] for d in data])
stds = np.array([d[1] for d in data])
plt.plot(temps[:-1], np.diff(aves)/np.diff(temps)/400)
plt.show()
plt.plot(temps, mags)
plt.show()

np.savez('J0_h0_1e6warm_4e8run.npz', temps=temps, aves=aves, mags=mags, stds=stds)