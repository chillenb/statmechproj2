from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import argparse

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r"""
\usepackage{newtxtext}
\usepackage{newtxmath}
\renewcommand{\familydefault}{\rmdefault}

"""

mpl.rcParams['pgf.preamble'] = r"""
\usepackage{newtxtext}
\usepackage{newtxmath}
\renewcommand{\familydefault}{\rmdefault}

"""
mpl.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({
    "font.family": "serif"
})

def tmag(T,J1,J2):
    return (1-1/(np.sinh(2*J1/T)*(np.sinh(2*J2/T)))**2)**(1/8)

if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='npz file to plot')
    parser.add_argument('-o', '--output', type=str, help='output file name')
    args = parser.parse_args()
    npzfile = np.load(args.filename)
    temps = npzfile['temps']
    aves = npzfile['aves']
    mags = npzfile['mags']
    
    tc = 2 / np.log(1+np.sqrt(2))

    log_grid_before_tc = tc - np.logspace(-60, np.log10(tc), num=1000, base=10)

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, layout='constrained', figsize=(7, 3.5))
    axs[0].plot(temps, mags, marker='+', fillstyle='none', markersize=3, color='black', linewidth=0.5, markeredgewidth=0.5, label='MC')
    axs[0].plot(log_grid_before_tc, tmag(log_grid_before_tc,1,1), color='blue', linestyle='dashed', linewidth=0.5, label='Theoretical')
    axs[0].set_ylabel("Spontaneous magnetization per spin")
    axs[0].legend()
    axs[0].set_xlabel(r'$k_B T / J$')
    

    axs[1].set_ylabel("Heat capacity per spin")
    axs[1].set_xlabel(r'$k_B T / J$')
    axs[1].plot(temps[:-1], np.diff(aves) / (400*np.diff(temps)), marker='o', fillstyle='none', markersize=3, color='black', linewidth=0.5, markeredgewidth=0.5)
    plt.savefig(args.output, backend='pgf')
