import os
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import tensorflow as tf
from util.input_data import load_data
from util.utils import preprocess_graph
from arga.link_reconstruction import Link_rec_Runner
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
import scipy.sparse as sp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('-d','--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'polblogs'], help='dataset')
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# make sure you use the same data splits as you generated attacks
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
tf.reset_default_graph()
tf.set_random_seed(args.seed)
###########################
# global settings
dataname = args.dataset     # 'cora' or 'citeseer' or 'polblogs'

if dataname in ['cora','citeseer','pubmed']:
    adj, features, idx_train, idx_val, idx_test, labels = load_data(dataname)
if dataname == "polblogs":
    data = Dataset(root='./data/', name=dataname)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


###########################
# vae settings
model = 'arga_vae'  #'arga_ae' or 'arga_vae'
iterations = 250
train = True  # train or load pretrain

###########################

sparse_search = list(range(1,40))
sparse_rate = [adj.sum() / adj.shape[0]**2 * r for r in sparse_search]
percentiles = [(1-r) * 100 for r in sparse_rate]
reconstructor = Link_rec_Runner(adj, features, model, iterations)
reconstructor.erun(train=train)

print("finishing reconstruction")

result = []
for p in percentiles:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    print(p)
    adj_rec = reconstructor.sparse(p)
    adj_rec_norm = preprocess_graph(adj_rec)
    model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
    model = model.to(device)
    model.fit(features, adj_rec_norm, labels, idx_train, idx_val, train_iters=200, verbose=False, normalize=False)
    model.eval()
    acc = model.test(idx_test)
    result.append(acc.item())
print(result)







