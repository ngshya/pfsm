# https://bergvca.github.io/2017/10/14/super-fast-string-matching.html


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
