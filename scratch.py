import ising
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import psutil


temps = np.flip(np.hstack([np.linspace(2,2.5, 32)]))
betas = 1/temps

def warmup(mod, beta):
    mod.random_mc_meanstd(beta, 1000000, 10000, verbose=True)

aves = np.zeros_like(betas)
stds = np.zeros_like(betas)

def measurebeta(b):
    m = ising.ModelWrap()
    warmup(m, b)
    mean = m.random_mc_large(b, 100000000, 100, verbose=True)
    return mean

ncpus = psutil.cpu_count(logical=False)
if not ncpus:
    ncpus = mp.cpu_count()
with mp.Pool(ncpus) as pool:
    aves = pool.map(measurebeta, list(betas))


plt.plot(temps[:-1], np.diff(aves)/np.diff(temps)/400)
plt.show()