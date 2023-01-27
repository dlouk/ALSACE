# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:55:45 2018

@author: Dimitris Loukrezis

Test ALSACE algorithm on meromorphic function 1/(G*Y), where
G = [g1, g2, ..., g16 = ][1, 0.5, 0.1, 0.05, 0.01, ..., 5*1e-8]
Y = [Y1, Y2, ..., Y16], Y_n ~ U[-1,1]
"""

import numpy as np
import openturns as ot
from alsace_Kopt import alsace
from tools import get_ed, compute_moments, PCE_Surrogate


def qoi_mero(yvec):
    """Meromorphic function"""
    gvec_tilde = np.array([1e0, 5*1e-1, 1e-1, 5*1e-2, 1e-2, 5*1e-3, 1e-3,
                           5*1e-4, 1e-4, 5*1e-5, 1e-5, 5*1e-6, 1e-6, 5*1e-7,
                           1e-7, 5*1e-8])
    coeff = 1.0 / (2.0*np.linalg.norm(gvec_tilde, ord=1))
    gvec = gvec_tilde * coeff
    dotprod = np.dot(gvec, yvec)
    return 1.0/(1 + dotprod)

# number of RVs
N = 16

# construct joint pdf
z = []
for i in range(N):
    if i%2==0:
        z.append(ot.TruncatedNormal(0,1,0,3))
    else:
        z.append(ot.TruncatedNormal(0,1,-3,0))
jpdf = ot.ComposedDistribution(z)

# generate cross-validation set
Ncv = 1000
ot.RandomGenerator.SetSeed(13)
cv_test_points, cv_values = get_ed(qoi_mero, jpdf, Ncv, 'R')

### results storage
meanz = []
varz = []
err_cv_max_m = []
err_cv_mean_m = []
err_cv_rms_m = []
fevalz = []
cond_numz = []
indicatorz = []
cardz = []
# vector of maximum function calls
max_fcalls = np.linspace(100, 1000, 10).tolist()

### APPROXIMATE
results_dict_m = {}
for mfc in max_fcalls:
    # simulation budget
    mfc = int(mfc)

    # pce dictionary
    results_dict_m = alsace(func=qoi_mero, N=N, jpdf=jpdf, max_fcalls=mfc,
                            limit_cond=100, sample_type='R',
                            pce_dict=results_dict_m, verbose=True)

    # collect all multi-indices and coefficients
    idx_all = results_dict_m['idx_act'] + results_dict_m['idx_adm']
    pce_coeff_all = results_dict_m['pce_coeff_act'] + results_dict_m['pce_coeff_adm']

    # compute pce surrogate model
    sur_model = PCE_Surrogate(pce_coeff_all, idx_all, jpdf)

    # compute moments
    mu, sigma2 = compute_moments(pce_coeff_all)
    meanz.append(mu)
    varz.append(sigma2)

    # evaluate PCE surrogate on cross-validation sample
    Y = sur_model.evaluate(cv_test_points)

    # compute cross-validation errors
    errs = np.abs(cv_values-np.reshape(np.asarray(Y), np.asarray(Y).shape[0]))
    err_cv_max_m.append(np.max(errs))
    err_cv_mean_m.append(np.mean(errs))
    err_cv_rms_m.append(np.sqrt(np.sum(errs**2)/len(errs)))

    # model evaluations, basis cardinality, global error indicators
    fevalz.append(len(results_dict_m['ed_fevals']))
    cardz.append(len(idx_all))
    indicatorz.append(np.sum(np.array(results_dict_m['pce_coeff_adm'])**2))

    print('fcalls:' + str(fevalz[-1]))
    print('pce terms:' + str(cardz[-1]))
    print('max cv error:'  + str(err_cv_max_m[-1]))
    print('mean cv error:'  + str(err_cv_mean_m[-1]))
    print('rms cv error:'  + str(err_cv_rms_m[-1]))
    print('global error indicator:'  + str(indicatorz[-1]))
    print("")
    results_all = np.array([fevalz, cardz, err_cv_max_m, err_cv_mean_m,
                            err_cv_rms_m]).T

#np.savetxt('alsace_meromorhic_trAinv_1.txt', results_all)
