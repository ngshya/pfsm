def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))



import re
import unidecode

def ngrams(string, n=3):
    string = string.upper()
    string = unidecode.unidecode(string)
    # string = re.sub(r'[,-./]|\sBD',r'', string)
    
    substitutions = [
        ('[^a-zA-Z0-9 ]', ''),
        ('\s+', ' '),
        ('IN LIQ[^ ]*|LIQUIDAZIONE|IN FALL[^ ]*|FALLIMENTO|FALL [0-9]+|FALL[0-9]+', ''),
        ("SOC.*COOP.*A.*R.*L.*", "SOCCOOPARL"),
        ("SOCIETA A RESPONSABILITA LIMITATA|SC A RL|S R L", "SRL"),
        (" S RL", " SRL"),
        ("SOCIETA COOPERATIVA SOCIALE", "SCS"),
        ("SOCIETA COOPERATIVA", "SC"),
        (" A RL| A R L| AR L", " ARL"),
        ("IMMOBILIARE", "IMM."),
        ("CONSORZIO", "CONS."),
        ("COOPERATIVA", "COOP."),
        ("COSTRUZIONI", "COST."),
        ("AUTOTRASPORTI", "AUT."),
        ("CENTRO", "CNT."),
        ("CONSORZIO", "CNS."),
        ("BANK OF", "B."),
        ("TELECOMMUNICATIONS", "TEL."), 
        ("CORPORATION", "CORP."), 
        ("TECHNOLOGIES", "TEC."),
        ("SERVICES", "SER."),
        ("PARTNERS", "PAR."),
        ("REAL ESTATE", "RE."),
        ("HOLDING", "H."),
        ("LABORATORIES", "LAB."),
        ("ENERGY", "EN."),
        ("CAPITAL", "CAP."),
        ("FACILITY", "FAC."),
        ("PROPERTIES", "PRP."), 
        ("ASSOCIATES", "ASS."),
        ("LIMITED", "LTD"),
        ("INVESTMENTS", "INV."),
        ("FUNDS TRUST", "FT."), 
        ("LIFE INSURANCE", "LI."), 
        ("OPPORTUNITIES", "OPP."), 
        ('\s+', ' ')
    ]
    
    for search, replacement in substitutions:
        string = re.sub(search, replacement, string)
        
    lst_split = string.split(" ")
    string = string + " " + lst_split[0] * 3
    if len(lst_split) > 1:
        string = string + " " + lst_split[1] * 2
    if len(lst_split) > 2:
        string = string + " " + lst_split[1] * 1
        
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]



from sklearn.feature_extraction.text import TfidfVectorizer

def tfidfM(array_names):
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(array_names)
    return tf_idf_matrix



import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))



def mycf(idxs, top_n, sim_th, ns):
    min_idxs = min(idxs)
    matches_tmp = awesome_cossim_top(ns.M[idxs, ], ns.M[min_idxs:,].transpose(), top_n, sim_th)
    return [matches_tmp[j, ].indices + min_idxs for j in range(matches_tmp.shape[0])]


import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import functools
import sys

def pfsm(strings, top_n = 10, sim_th = 0.85, n_splits = 500, n_proc = 10):
    
    strings = np.unique(strings)
    
    if len(strings) < n_splits:
        n_splits = len(strings)
            
    M = tfidfM(strings)
    
    int_entity_id = np.array([np.NaN for j in range(len(strings))])
    lst_split = list(split(range(M.shape[0]), n_splits))
    
    manager = multiprocessing.Manager()
    ns = manager.Namespace()
    ns.M = M
    
    pool = Pool(n_proc)
    r = pool.map(functools.partial(mycf, top_n = top_n, sim_th = sim_th, ns = ns), lst_split)
    pool.close()
    pool.join()
        
    counter = 0

    for k in range(len(r)):
        tmp_array = r[k]
        for j in range(len(tmp_array)):
            idx = tmp_array[j]
            if np.isscalar(idx):
                idx = np.array([idx])
            idx = idx.astype(int)
            tmp_idx = int_entity_id[idx]
            if sum([np.isnan(x) for x in tmp_idx]) == len(tmp_idx):
                int_entity_id[idx] = counter
            else:
                int_entity_id[idx] = np.nanmin(tmp_idx)
            counter = +1

    dtf_out = pd.DataFrame({"STRING": strings, "ENTITY_ID": int_entity_id}).sort_values(["ENTITY_ID"])
    
    return dtf_out