import ising
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import psutil


temps = np.flip(np.hstack([np.linspace(0.01,4, 30)]))
betas = 1/temps

def warmup(mod, beta):
    mod.random_mc_meanstd(beta, 100000000, 10000, verbose=False)

aves = np.zeros_like(betas)
stds = np.zeros_like(betas)

def measurebeta(b, seed):
    m = ising.ModelWrap(seed=seed)
    warmup(m, b)
    #mean, std = m.random_mc_meanstd(b, 400000000, 1, verbose=True, chunksize=10000000)
    #mean, std = m.random_mc_meanstd(b, 2000000000, 1, verbose=True)
    mean, std = m.random_mc_meanstd(b, 600000000, 1, verbose=True)

    return mean

ncpus = psutil.cpu_count(logical=True)
if not ncpus:
    ncpus = mp.cpu_count()
with mp.Pool(ncpus) as pool:
    seq = np.random.SeedSequence(12345)
    ss = seq.spawn(len(betas))
    aves = pool.starmap(measurebeta, zip(list(betas), ss))
#measurebeta(betas[0])


plt.plot(temps[:-1], np.diff(aves)/np.diff(temps)/400)
plt.show()