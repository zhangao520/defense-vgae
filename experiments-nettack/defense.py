import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import warnings
warnings.filterwarnings('ignore')
import sys
import torch
sys.path.append('..')

from nettack import nettack, GCN
from util.input_data import load_data
from util import utils
from arga.link_reconstruction import Link_rec_Runner
from deeprobust.graph.defense import GCN as GCNTorch
from deeprobust.graph.defense import  GCNJaccard, GCNSVD, RGCN
from deeprobust.graph.data import Dataset
import numpy as np
import logging
from baseline import jaccard

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('-t', '--target', type=int, help="target node", default=2000)
parser.add_argument('-d','--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'polblogs'], help='dataset')
parser.add_argument('-j', '--threshold', type=float, help="Jaccard model's threshold", default=0.05)
parser.add_argument('-p', '--perturb', type=int, help="nettack perturb", default=1)
parser.add_argument('-r', '--rank', type=int, help="SVD model's rank.", default=10)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###########################
# global settings
# 'cora' or 'citeseer' or 'polblogs'
dataname = args.dataset
gpu_id = '0'

if dataname in ['cora','citeseer','pubmed']:
    adj, features, idx_train, idx_val, idx_test, labels = load_data(dataname,root='..')
if dataname == "polblogs":
    data = Dataset(root='../data/', name=dataname)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
num_class = labels.max()+1
adj_norm = utils.preprocess_graph(adj)
labels_onehot = np.eye(num_class)[labels]
preds={}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################## attacking ##################################
degrees = adj.sum(0).A1
hidden_dim = 16
direct_attack = True
n_influencers = 1 if direct_attack else 5

u = args.target
preds['gt'] = labels[u] 
n_perturbations = args.perturb
perturb_features = False
perturb_structure = True


surrogate_model = GCN.GCN([hidden_dim, num_class], adj_norm, features, 
            with_relu=False, name="surrogate", gpu_id=gpu_id,seed=args.seed)
surrogate_model.train(idx_train, idx_val, labels_onehot)
W1 =surrogate_model.W1.eval(session=surrogate_model.session)
W2 =surrogate_model.W2.eval(session=surrogate_model.session)

nettack = nettack.Nettack(adj, features, labels, W1, W2, u, verbose=True)
nettack.attack_surrogate(n_perturbations, perturb_structure=perturb_structure, 
                        perturb_features=perturb_features, direct=direct_attack, n_influencers=n_influencers)

print(nettack.structure_perturbations)


###########################
# no defense
model = GCNTorch(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)
model.fit(features, nettack.adj_preprocessed, labels, idx_train, idx_val, train_iters=200, verbose=False, normalize=False)
model.eval()
acc = model.test(idx_test)
probs_after_defense = model.predict().detach().cpu().numpy()[u]
preds['no'] = np.argmax(probs_after_defense)


###########################
# vae settings
model = 'arga_vae'  #'arga_ae' or 'arga_vae'
iterations = 500 if dataname=="polblogs" else 200
train = True  # train or load pretrain

###########################

percentile = (1-adj.sum() / adj.shape[0]**2 * 20) * 100
reconstructor = Link_rec_Runner(nettack.adj, features, model, iterations)
reconstructor.erun(train=train)
print("finishing reconstruction")

###########################
# Ours
adj_rec = reconstructor.sparse(percentile)
adj_rec_norm = utils.preprocess_graph(adj_rec)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
model = GCNTorch(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)
model.fit(features, adj_rec_norm, labels, idx_train, idx_val, train_iters=200, verbose=False, normalize=False)
model.eval()
acc = model.test(idx_test)
probs_after_defense = model.predict().detach().cpu().numpy()[u]
preds['ours'] = np.argmax(probs_after_defense)

###########################
# Ours + Jaccard
if dataname=='polblogs':
    preds['jaccard'] = -1
else:
###########################
# Jaccard 
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    model = GCNJaccard(nfeat=features.shape[1], nclass=num_class,
                    nhid=16, device=device)

    model = model.to(device)

    print('=== testing GCN-Jaccard on perturbed graph ===')
    model.fit(features, nettack.adj.tocsr(), labels, idx_train, idx_val, threshold=args.threshold, verbose=False)
    model.eval()
    acc = model.test(idx_test)
    probs_after_defense = model.predict().detach().cpu().numpy()[u]
    preds['jaccard'] = np.argmax(probs_after_defense)

if dataname=='polblogs':
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    model = GCNSVD(nfeat=features.shape[1], nclass=labels.max()+1,
                    nhid=16, device=device)
    model = model.to(device)
    model.fit(features, nettack.adj.tocsr(), labels, idx_train, idx_val, k=args.rank, verbose=True)
    model.eval()
    output = model.test(idx_test)
    probs_after_defense = model.predict().detach().cpu().numpy()[u]
    preds['svd'] = np.argmax(probs_after_defense)
else:
    preds['svd'] = -1

###########################
# RGCN
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
model = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=num_class,
                nhid=32, device=device)
model = model.to(device)
model.fit(features, nettack.adj.tocsr(), labels, idx_train, idx_val, train_iters=200, verbose=False)
model.eval()
# You can use the inner function of model to test
model.test(idx_test)
probs_after_defense = model.output[u].detach().cpu().numpy()
preds['rgcn'] = np.argmax(probs_after_defense)


logging.basicConfig(level=logging.CRITICAL, filename='log/defense-jaccard_%.2f-svd_%d-%s-%dp.log'
                    %(args.threshold, args.rank, dataname,n_perturbations), filemode='a')
logging.critical("target node: %d\tgt: %d\tno: %d\tours: %d\tjaccard: %d\tsvd: %d\trgcn: %d"
        %(u, labels[u], preds['no'], preds['ours'], preds['jaccard'],preds['svd'],preds['rgcn']
        )) 