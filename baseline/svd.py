import scipy.sparse as sp
import numpy as np

def truncatedSVD(data, k=10):
    print(f'=== GCN-SVD: rank={k} ===')
    if sp.issparse(data):
        data = data.asfptype()
        U, S, V = sp.linalg.svds(data, k=k)
        print(f"rank_after = {len(S.nonzero()[0])}")
        diag_S = np.diag(S)
    else:
        U, S, V = np.linalg.svd(data)
        U = U[:, :k]
        S = S[:k]
        V = V[:k, :]
        print(f"rank_before = {len(S.nonzero()[0])}")
        diag_S = np.diag(S)
        print(f"rank_after = {len(diag_S.nonzero()[0])}")
    res = U @ diag_S @ V
    res = sp.csr_matrix(res)
    
    return res

