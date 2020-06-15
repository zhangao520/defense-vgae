import scipy.sparse as sp
import numpy as np
import tqdm

# DeepRobust implementation
def drop_dissimilar_edges(adj, features, threshold=0.01):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    modified_adj = adj.copy().tolil()
    # preprocessing based on features
    print(modified_adj.shape)

    print('=== GCN-Jaccrad ===')
    edges = np.array(modified_adj.nonzero()).T
    removed_cnt = 0
    for edge in tqdm.tqdm(edges):
        n1 = edge[0]
        n2 = edge[1]
        if n1 > n2:
            continue

        J = _jaccard_similarity(features[n1], features[n2])

        if J < threshold:
            modified_adj[n1, n2] = 0
            modified_adj[n2, n1] = 0
            removed_cnt += 1

    print(f'removed {removed_cnt} edges in the original graph')
    return modified_adj

def _jaccard_similarity(a, b):
    intersection = a.multiply(b).count_nonzero()
    J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
    return J