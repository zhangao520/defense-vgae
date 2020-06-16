import os
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
sys.path.append('..')
import tensorflow as tf
import scipy.sparse as sp
import torch
import numpy as np
from arga.link_reconstruction import Link_rec_Runner
from util import utils
from util.input_data import load_data
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('-d', '--dataset', type=str, default='polblogs', choices=['cora', 'citeseer', 'polblogs'], help='dataset')
parser.add_argument('-p', '--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('-r', '--rank',type=int, default=10, help="SVD model's rank")
parser.add_argument('-j', '--threshold', type=float, help="Jaccard model's threshold", default=0.02)

args = parser.parse_args()

args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
tf.reset_default_graph()
tf.set_random_seed(args.seed)
dataname = args.dataset
ptb_rate = args.ptb_rate

data_setting = 'nettack' if dataname == 'polblogs' else 'gcn'

if dataname in ['cora','citeseer','pubmed']:
    adj, features, idx_train, idx_val, idx_test, labels = load_data(dataname, root='..')
if dataname == "polblogs":
    data = Dataset(root='../data/', name=dataname)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


perturbed_data_filename = "tmp/%s_mettack_%.2f.npz"%(dataname, ptb_rate)
perturbed_adj = sp.load_npz(perturbed_data_filename)

assert perturbed_adj.shape == adj.shape
###########################
# vae setting 
model = 'arga_vae'  #'arga_ae' or 'arga_vae'
iterations = 500
train = True   # train or load pretrain

reconstructor = Link_rec_Runner(perturbed_adj, features, model, iterations)
reconstructor.erun(train=train)
print("finishing reconstruction")


print("searching for sparsity...")
res = []
for sr in range(1,40):
    percentile = (1-perturbed_adj.sum() / adj.shape[0]**2 * sr) * 100
    adj_rec = reconstructor.sparse(percentile)
    adj_rec_norm = utils.preprocess_graph(adj_rec)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    vgae = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
    vgae= vgae.to(device)
    vgae.fit(features, adj_rec_norm, labels, idx_train, idx_val, train_iters=200, verbose=False, normalize=False)
    vgae.eval()
    output = vgae.test(idx_val)
    res.append(output.item())


print("retraining")
bst = np.argmax(res) + 1
percentile = (1-perturbed_adj.sum() / adj.shape[0]**2 * bst) * 100
adj_rec = reconstructor.sparse(percentile)
adj_rec_norm = utils.preprocess_graph(adj_rec)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
vgae = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
vgae= vgae.to(device)
vgae.fit(features, adj_rec_norm, labels, idx_train, idx_val, train_iters=200, verbose=False, normalize=False)
vgae.eval()
output = vgae.test(idx_test)