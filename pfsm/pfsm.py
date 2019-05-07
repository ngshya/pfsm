from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import numpy as np

def cosimtop(A, B, ntop, lower_bound=0):
    '''
    Optimized cosine similarity computation.
    :param A: First matrix.
    :param B: Second matrix.
    :param ntop: Top n for each row.
    :param lower_bound: Lower bound for each row.
    :return: Cosine similarity matrix.
    '''
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
    idx_dtype = np.int32
    nnz_max = M * ntop
    indptr = np.zeros(M + 1, dtype = idx_dtype)
    indices = np.zeros(nnz_max, dtype = idx_dtype)
    data = np.zeros(nnz_max, dtype = A.dtype)
    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype = idx_dtype),
        np.asarray(A.indices, dtype = idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype = idx_dtype),
        np.asarray(B.indices, dtype = idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)
    return csr_matrix((data, indices, indptr), shape=(M, N))

# # # # # # # # # # # # # # # # # #

def getEntityIdInsideGroup(inputs):
    '''
    Get entity ids.
    :param inputs: Input data.
    :return: Dictionry containing IDs.
    '''
    M = inputs["M"]
    idx = inputs["idx"]
    matches_tmp = cosimtop(M, M.transpose(), inputs["cosimtop_ntop"], inputs["cosimtop_lower_bound"])
    out = np.array([min(matches_tmp[j,].indices, default = j) for j in range(matches_tmp.shape[0])])
    for index, value in enumerate(out):
        if out[value] != value:
            out[index] = out[value]
    return {i: x for i, x in zip(idx, out)}

# # # # # # # # # # # # # # # # # #

import numpy as np
import pandas as pd

import unidecode
import re
import time
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

import multiprocessing

class Strings:

    def __init__(self, str_strings = [""]):
        '''
        Initialization.
        :param str_strings: Array of strings to be compared.
        '''
        self.original_strings = np.array(str_strings)

    # # # # # # # # # # # # # # # # # #

    def clean(self, string):
        '''
        Strings cleaning method. Overwrite with custom one.
        :param string: String to be cleanned.
        :return: Cleanned string.
        '''
        string = string.upper()
        string = unidecode.unidecode(string)
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
            ("AGENZIA IMMOBILIARE", "IMM."),
            ("AGENZIA DI ASSICURAZION", "AA."),
            ("AGENZIA VIAGGI", "AV."),
            ("ACLI SERVICE", "AS."),
            ("ACCADEMIA DI BELLE ARTI", "AB."),
            ("ACCADEMIA", "AC."),
            ("DISTRIBUZIONE", "DIS."),
            ("SERVIZI", "SRV."),
            ("SOCIETA AGRICOLA", "SA."),
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
            ("CONSULTING", "CST"),
            ("SERVICE", "SRV"),
            ("LLC", ""),
            ("^1", ""),
            ('\s+', ' ')
        ]
        for search, replacement in substitutions:
            string = re.sub(search, replacement, string)
        return string[0:100]

    # # # # # # # # # # # # # # # # # #

    def enrich(self, string, min_chars = 3):
        '''
        Strings enrichment method. Overwrite with custom one.
        :param string: Input string.
        :param min_chars: Minimum number of characters..
        :return: Enriched string.
        '''
        if len(string) < min_chars:
            string = string + " " + ("#" * min_chars)

        lst_split = string.split(" ")
        string = string + " " + lst_split[0] * 3
        if len(lst_split) > 1:
            string = string + " " + lst_split[1] * 2
        if len(lst_split) > 2:
            string = string + " " + lst_split[1] * 1
        return string

    # # # # # # # # # # # # # # # # # #

    def processStrings(self, min_chars = 3):
        '''
        Clean and enrich the strings.
        :param min_chars: Minimum number of characters.
        :return: Processed string.
        '''
        self.processed_strings = np.array([self.enrich(self.clean(x), min_chars = min_chars) for x in self.original_strings])

    # # # # # # # # # # # # # # # # # #

    def ngrams(self, string, n = 3):
        '''
        Split a string in ngrams.
        :param string: String to be splitted.
        :param n: n of n-grams.
        :return: Array of ngrams.
        '''
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    # # # # # # # # # # # # # # # # # #

    def tfidfM(self, strings = None, n = 3):
        '''
        Create the tfidf matrix for the strings.
        :param n: n for ngrams.
        :return: Sparse tfidf matrix.
        '''
        vectorizer = TfidfVectorizer(min_df = 1, analyzer = lambda x: self.ngrams(x, n = n))
        if strings is None:
            tf_idf_matrix = vectorizer.fit_transform(self.processed_strings)
            self.M = tf_idf_matrix
            self.N = self.M.copy()
            self.N.data = (self.N.data > 0) + 0
            self.Ncsum = np.array(self.N.sum(axis = 0))[0]
            self.NcsumSorted = sorted(self.Ncsum, reverse = True)
        else:
            tf_idf_matrix = vectorizer.fit_transform(strings)
            return tf_idf_matrix

    # # # # # # # # # # # # # # # # # #

    def hGroupByKMeans(self, idx, ns, id = None):
        '''
        Group by using KMeans recursively.
        :param idx: Indices array.
        :param ns: Namespace of a manager.
        :return: A dataframe containing group IDs.
        '''
        if id is None:
            id = self.groups_ids[idx[0]]
        tmp_iter = len(id.split("-")) - 1
        lbidx = tmp_iter * ns["kmeans_ncols_step"]
        ubidx = (tmp_iter + 1) * ns["kmeans_ncols_step"]
        if ubidx >= len(self.NcsumSorted):
            ubidx = len(self.NcsumSorted) - 1
        if lbidx >= len(self.NcsumSorted):
            lbidx = max(len(self.NcsumSorted) - ns["kmeans_ncols_step"], 0)
        tmp_min = self.NcsumSorted[ubidx]
        tmp_max = self.NcsumSorted[lbidx]
        if tmp_max <= 10:
            tmp_max = self.NcsumSorted[0]
        bln_tmp = ((tmp_max >= self.Ncsum) & (self.Ncsum >= tmp_min))
        M = self.M[:, bln_tmp]
        M = M[idx, ]
        kmeans = MiniBatchKMeans(
            n_clusters = ns["kmeans_n_clusters"],
            max_iter = ns["kmeans_max_iter"],
            batch_size = ns["kmeans_batch_size"],
            random_state = 1102
        ).fit(M)
        new_ids = kmeans.labels_.astype(str)
        self.groups_ids.update({i: id + "-" + k for i, k in zip(idx, new_ids)})

    # # # # # # # # # # # # # # # # # #

    def groupByKMeans(
        self,
        kmeans_ncols_step = 12,
        kmeans_max_dim_cluster = 30000,
        kmeans_n_clusters = 128,
        kmeans_max_iter = 32,
        kmeans_batch_size = 512,
        kmeans_n_jobs = -1
    ):
        '''
        Wrapper for recurGroupByKMeans()
        :param kmeans_ncols_step: Step for kmeans_ncols.
        :param kmeans_max_dim_cluster: Maximum dimension of each cluster.
        :param kmeans_n_clusters: Number of cluster for each iteration of KMeans.
        :param kmeans_max_iter: KMeans maximum number of iterations.
        :param kmeans_batch_size: MiniBatchKMeans batch size.
        :param kmeans_n_jobs: KMeans number of jobs.
        :return: Dataframe containing groups IDs.
        '''
        ns = dict()
        ns["kmeans_max_dim_cluster"] = kmeans_max_dim_cluster
        ns["kmeans_n_clusters"] = kmeans_n_clusters
        ns["kmeans_ncols_step"] = kmeans_ncols_step
        ns["kmeans_max_iter"] = kmeans_max_iter
        ns["kmeans_batch_size"] = kmeans_batch_size
        ns["kmeans_n_jobs"] = kmeans_n_jobs
        self.groups_ids = {j: "0" for j in range(len(self.processed_strings))}

        groups_ids_counts = np.unique(list(self.groups_ids.values()), return_counts = True)
        max_cluster_dim = max(groups_ids_counts[1])
        max_cluster_id = np.array(groups_ids_counts[0])[np.array(groups_ids_counts[1]) == max_cluster_dim][0]

        while max_cluster_dim > ns["kmeans_max_dim_cluster"]:
            self.echo("Largest cluster dim: " + str(max_cluster_dim) + " | Largest cluster ID " + str(max_cluster_id))
            tms_start = time.time()
            idx = np.array(list(self.groups_ids.keys()))[np.array(list(self.groups_ids.values())) == max_cluster_id]
            self.hGroupByKMeans(idx, ns, max_cluster_id)
            self.echo("Largest cluster dim: " + str(max_cluster_dim) + " | Largest cluster ID " + str(
                max_cluster_id) + " | Iteration time: " + str(int(time.time() - tms_start)))
            groups_ids_counts = np.unique(list(self.groups_ids.values()), return_counts=True)
            max_cluster_dim = max(groups_ids_counts[1])
            max_cluster_id = np.array(groups_ids_counts[0])[np.array(groups_ids_counts[1]) == max_cluster_dim][0]

    # # # # # # # # # # # # # # # # # #

    def getEntityID(self, n_jobs = 12, ntop = 5, lb = 0.88):
        '''
        Get entity IDs for each group.
        :param n_jobs: Number of jobs.
        :return: Dataframe containing entity IDs.
        '''
        self.index_strings = np.array(list(self.groups_ids.keys()))
        print("\nCreating array of clusters.")
        tmp_dict = {}
        for key, value in self.groups_ids.items():
            if value in tmp_dict.keys():
                tmp_dict[value].append(key)
            else:
                tmp_dict[value] = [key]
        self.shared_array_entities_ids = {}
        print("There are " + str(len(tmp_dict)) + " clusters." + " | Input dictionary size: " + str(sys.getsizeof(tmp_dict)))
        pool = multiprocessing.Pool(n_jobs)
        r = pool.map(getEntityIdInsideGroup, [{"idx": idx, "M": self.M[idx, ], "cosimtop_ntop": ntop, "cosimtop_lower_bound": lb} for idx in list(tmp_dict.values())])
        pool.close()
        pool.join()
        print("Multiprocessing done! Merging data ...")
        for x in r:
            self.shared_array_entities_ids.update(x)
        self.groups = pd.DataFrame({"INDEX": list(self.shared_array_entities_ids.keys()),
                                    "ENTITY_ID": list(self.shared_array_entities_ids.values())})
        self.groups = self.groups.drop_duplicates(["INDEX"])
        self.groups = pd.merge(
            pd.DataFrame({"ORIGINAL_STRING": self.original_strings,
                          "INDEX": self.index_strings,
                          "GROUP_ID": np.array(list(self.groups_ids.values()))}),
            self.groups,
            how = "left",
            on = "INDEX"
        )
        self.groups.ENTITY_ID = [x + "-" + str(y) for x, y in zip(self.groups.GROUP_ID, self.groups.ENTITY_ID)]

    # # # # # # # # # # # # # # # # # #

    def echo(self, x):
        sys.stdout.write('\r' + str(x))
        sys.stdout.flush()





