# -*- coding: utf-8 -*-
"""
Test performance of gap statistic on normal data.
@author: ysirotin
"""

import numpy as np
import scipy as sc
from gap_statistic import OptimalK
import matplotlib.pyplot as plt

optimalK = OptimalK(parallel_backend='rust')

a1 = 0
mu1 = 0.0
sig1 = 0.1

a2 = 0
mu2 = 1.0
sig2 = 0.1


N = 10000
x = np.linspace(-2,3,100)

fig, ax = plt.subplots(2, 2)
axl = ax.reshape(1,4)[0]
for ii in range(4):
    rv1 = sc.stats.skewnorm(a1, loc=mu1, scale=(ii+1)*sig1)
    rv2 = sc.stats.skewnorm(a2, loc=mu2, scale=(ii+1)*sig2)

    X = np.reshape(np.concatenate((rv1.rvs(size=N), rv2.rvs(size=N))),(2*N,1))
    nclusters = optimalK(X, cluster_array=np.arange(1,5))
    gap_value = optimalK.gap_df['gap_value'][optimalK.gap_df['n_clusters']==nclusters]
    dprime = (mu2-mu1)/np.sqrt(0.5 * (((ii+1)*sig1)**2+((ii+1)*sig2)**2))

    axl[ii].hist(X, 100, linewidth=0, normed=True)
    axl[ii].plot(x, rv1.pdf(x)/2)
    axl[ii].plot(x, rv2.pdf(x)/2)
    axl[ii].set_title('Optimal Clusters = %d, \nGap = %1.2f, d\' = %1.2f' % (nclusters, gap_value, dprime))

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig(filename = 'test_gap.png', dpi = 96)