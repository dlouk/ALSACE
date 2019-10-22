
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:43:11 2018
@author: galetzka
"""
import openturns as ot
import numpy as np
import scipy.linalg as sp
from idx_admissibility import admissible_neighbors
from tools import get_ed, transform_multi_index_set, get_design_matrix


def alsace(func, N, jpdf, tol=1e-22, sample_type='R', limit_cond=100,  
           max_fcalls=1000, seed=123, ed_file=None, ed_fevals_file=None, 
           verbose=True, pce_dict={}):
    """
    ALSACE - Approximations via Lower-Set and Least-Squares-based Adaptive Chaos Expansions
    
    func: function to be approximated.
    N: number of parameters.
    jpdf: joint probability density function.
    limit_cond: maximum allowed condition number of design matrix D
    sample_type: 'R'-random, 'L'-LHS
    seed: sampling seed
    tol, max_fcalls: exit criteria, self-explanatory.
    ed_file, ed_fevals_file: experimental design and corresponding evaluations
    
    'act': activated, i.e. already part of the approximation.
    'adm': admissible, i.e. candidates for the approximation's expansion.
    """
    
    if not pce_dict: # if pce_dict is empty --> cold-start 
        idx_act = []
        idx_act.append([0]*N) # start with 0 multi-index
        idx_adm = []
        # set seed
        ot.RandomGenerator.SetSeed(seed) 
        ed_size = 2*N # initial number of samples
        # initial experimental design and coresponding evaluations
        ed, ed_fevals = get_ed(func, jpdf, ed_size, sample_type=sample_type, 
                               knots=[], values=[], ed_file=ed_file, 
                               ed_fevals_file=ed_fevals_file)
        global_error_indicator = 1.0 # give arbitrary sufficiently large value 
        
        # get the distribution type of each random variable
        dist_types = []
        for i in range(N):
            dist_type = jpdf.getMarginal(i).getName()
            dist_types.append(dist_type)
            
        # create orthogonal univariate bases
        poly_collection = ot.PolynomialFamilyCollection(N)
        for i in range(N):
            if dist_types[i] == 'Uniform':
                poly_collection[i] = ot.OrthogonalUniVariatePolynomialFamily(ot.LegendreFactory()) 
            elif dist_types[i] == 'Normal':
                poly_collection[i] = ot.OrthogonalUniVariatePolynomialFamily(ot.HermiteFactory()) 
            elif dist_types[i] == 'Beta':
                poly_collection[i] = ot.OrthogonalUniVariatePolynomialFamily(ot.JacobiFactory())
            elif dist_types[i] == 'Gamma':
                poly_collection[i] = ot.OrthogonalUniVariatePolynomialFamily(ot.LaguerreFactory())        
            else:
                pdf = jpdf.getDistributionCollection()[i]
                algo = ot.AdaptiveStieltjesAlgorithm(pdf)
                poly_collection[i] = ot.StandardDistributionPolynomialFactory(algo)
                
        # create multivariate basis 
        mv_basis = ot.OrthogonalProductPolynomialFactory(poly_collection, 
                                                        ot.EnumerateFunction(N))
        # get enumerate function (multi-index handling)
        enum_func = mv_basis.getEnumerateFunction()        
        
    else: # get data from dictionary
        idx_act = pce_dict['idx_act']
        idx_adm = pce_dict['idx_adm']
        pce_coeff_act = pce_dict['pce_coeff_act']
        pce_coeff_adm = pce_dict['pce_coeff_adm']
        ed = pce_dict['ed']
        ed_fevals = pce_dict['ed_fevals']
        ed_size = len(ed_fevals)
        # compute local and global error indicators
        global_error_indicator = np.sum(np.array(pce_coeff_adm)**2)
        enum_func = pce_dict['enum_func']
        mv_basis = pce_dict['mv_basis']
    
    #
    while ed_size < max_fcalls and global_error_indicator > tol:
        # the index added last to the activated set is the one to be refined
        last_act_idx = idx_act[-1][:]
        # get admissible neighbors of the lastly added index
        adm_neighbors = admissible_neighbors(last_act_idx, idx_act)
        # update admissible indices
        idx_adm = idx_adm + adm_neighbors
        # get polynomial basis for the LS problem
        idx_ls = idx_act + idx_adm
        idx_ls_single = transform_multi_index_set(idx_ls, enum_func)
        ls_basis = mv_basis.getSubBasis(idx_ls_single)
        ls_basis_size = len(ls_basis)
        
        # construct the design matrix D and compute its QR decomposition and its 
        # condition number
        D = get_design_matrix(ls_basis, ed)
        Q, R = sp.qr(D, mode='economic')
        condD =  np.linalg.cond(R)
        
        # If condD becomes too large, enrich the ED until condD becomes acceptable
        # or until ed_size reaches max_fcalls
        while (condD > limit_cond and ed_size < max_fcalls) or ed_size < ls_basis_size:
            # inform user
            if verbose: 
                print('WARNING: condition(D) = ' , condD)
                print("")
            # select new size for the ED
            if ls_basis_size > ed_size:
                ed_size = ls_basis_size + N
            elif ed_size + N > max_fcalls:
                ed_size = max_fcalls
            else:
                ed_size = ed_size + N
            # expand ED
            ed, ed_fevals = get_ed(func, jpdf, ed_size, sample_type=sample_type, 
                                   knots=ed, values=ed_fevals, ed_file=ed_file, 
                                   ed_fevals_file=ed_fevals_file)
            # construct the design matrix D and compute its QR decomposition and its 
            # condition number
            D = get_design_matrix(ls_basis, ed)
            Q, R = sp.qr(D,mode='economic')
            condD =  np.linalg.cond(R)
        
        # solve LS problem 
        c = Q.T.dot(ed_fevals)
        pce_coeff_ls = sp.solve_triangular(R, c)
            
        # find the multi-index with the largest contribution, add it to idx_act
        # and delete it from idx_adm
        pce_coeff_act = pce_coeff_ls[:len(idx_act)].tolist()
        pce_coeff_adm = pce_coeff_ls[-len(idx_adm):].tolist()
        help_idx = np.argmax(np.abs(pce_coeff_adm))
        idx_add = idx_adm.pop(help_idx)
        pce_coeff_add = pce_coeff_adm.pop(help_idx)
        idx_act.append(idx_add)
        pce_coeff_act.append(pce_coeff_add)

    # store expansion data in dictionary
    pce_dict = {}
    pce_dict['idx_act'] = idx_act
    pce_dict['idx_adm'] = idx_adm
    pce_dict['pce_coeff_act'] = pce_coeff_act
    pce_dict['pce_coeff_adm'] = pce_coeff_adm
    pce_dict['ed'] = ed
    pce_dict['ed_fevals'] = ed_fevals
    pce_dict['enum_func'] = enum_func
    pce_dict['mv_basis'] = mv_basis
    return pce_dict
    