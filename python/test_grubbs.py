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
mu = 0.
sig = 1.

# check if grubbs works as advertized
def test_grubbs(a=0, mu=0, sig=1, N=100, alpha = 0.05, N_reps = 10000):
    n_fa = 0
    for rep in range(N_reps):
        if len(grubbs.max_test_outliers(sc.stats.skewnorm.rvs(a,loc=mu,scale=sig,size=N),alpha=0.1)) > 0:
            n_fa+=1
            
    return n_fa/N_reps

