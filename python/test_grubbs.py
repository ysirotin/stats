# -*- coding: utf-8 -*-
"""
Test Grubb's Test Performance on Skewed Distributions
@author: ysirotin
"""

from outliers import smirnov_grubbs as grubbs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

# parameters
N = 100 # sample population size

a = 4
mu = 0.0
sig = 0.1

# check if grubbs works as advertized
def test_grubbs(rv, N=200, alpha = 0.05, N_reps = 1000):
    n_fa = 0
    val_fa = []
    for rep in range(N_reps):
        outliers = grubbs.max_test_outliers(rv.rvs(size=N), alpha=alpha)
        if len(outliers) > 0:
            val_fa+=outliers
            n_fa+=1
            
    return n_fa/N_reps, val_fa

fig, ax = plt.subplots(2, 2)
axl = ax.reshape(1,4)[0]
for ii in range(4):
    rv = sc.stats.skewnorm(ii, loc=mu, scale=sig)
    x = np.linspace(-0.4,0.4,100)
    bins = np.linspace(-0.4,0.4,50)
    pfa, val_fa = test_grubbs(rv)
    axl[ii].hist(val_fa, bins=bins, linewidth=0, normed=True)
    axl[ii].plot(x,rv.pdf(x))
    axl[ii].set_title('skew = %d, pfa = %1.2f' % (ii,np.around(pfa, decimals=2)))
    
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

fig.savefig(filename = 'test_grubbs.png', dpi = 96)