from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r"""
\usepackage{newtxtext}
\usepackage{newtxmath}
"""

def make_plot(tempgrid, eperspin):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, tight_layout=True)
    axs[0].plot(tempgrid, eperspin, marker='v', fillstyle='none', markersize=4, color='black', linewidth=0.5, markeredgewidth=0.5)
    axs[0].set_ylabel(r'$E(T)/N$')
    axs[0].set_xlabel(r'$k_B T / J$')

    axs[1].set_ylabel(r'$c(T)$')
    axs[1].set_xlabel(r'$k_B T / J$')
    axs[1].plot(tempgrid[:-1], np.diff(eperspin) / np.diff(tempgrid), marker='o', fillstyle='none', markersize=4, color='black', linewidth=0.5, markeredgewidth=0.5)
    plt.show()

if __name__ == '__main__':
    npzfile = np.load('heat.npz')
    grid = npzfile['grid']
    es = npzfile['es']
    eperspin = 0.5*np.mean(np.array(es), axis=1)/(64*64)
    make_plot(grid, eperspin)
