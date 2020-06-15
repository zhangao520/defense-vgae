from __future__ import division
from __future__ import print_function
import os
import numpy as np
import scipy.sparse as sp
# Train on CPU (hide GPU) due to memory constraints

import tensorflow as tf
from .constructor import get_placeholder, get_model,  get_optimizer, update, process_data, format_data
from .metrics import linkpred_metrics
# Settings


class Link_rec_Runner():
    def __init__(self, adj, features, model, iterations):
        self.adj = adj
        self.features = features
        self.iteration = iterations
        self.model = model
        #self.sparse_rate = sparse_rate
        #self.percentile = (1-sparse_rate) * 100

    def erun(self, train=True):
        model_str = self.model
        # formatted data
        feas = process_data(self.adj, self.features)
        # Define placeholders
        placeholders = get_placeholder(feas['adj'])

        # construct model
        d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])

        # Initialize session
        sess = tf.Session()
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # config = tf.ConfigProto(gpu_options=gpu_options)
        # sess = tf.InteractiveSession(config=config)

        sess.run(tf.global_variables_initializer())

        val_roc_score = []

        # Train model
        if train:
            for epoch in range(self.iteration):

                emb, construction, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])

                lm_train = linkpred_metrics(feas['val_edges'], feas['val_edges_false'])
                roc_curr, ap_curr, _ = lm_train.get_roc_score(emb, feas)
                val_roc_score.append(roc_curr)

                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost), "val_roc=", "{:.5f}".format(val_roc_score[-1]), "val_ap=", "{:.5f}".format(ap_curr))
                if (epoch+1) % 10 == 0:
                    ae_model.save(sess = sess)
        else:
            ae_model.load(sess = sess)
            emb, construction, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])

            lm_train = linkpred_metrics(feas['val_edges'], feas['val_edges_false'])
            roc_curr, ap_curr, _ = lm_train.get_roc_score(emb, feas)
            val_roc_score.append(roc_curr)

            print("train_loss=", "{:.5f}".format(avg_cost), "val_roc=", "{:.5f}".format(val_roc_score[-1]), "val_ap=", "{:.5f}".format(ap_curr))            

        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        #print(construction[feas['val_edges_false']])
        adj_rec = sigmoid(np.dot(emb, emb.T))
        #self.adj_rec = adj_rec - sp.dia_matrix((adj_rec.diagonal()[np.newaxis, :], [0]), shape=adj_rec.shape)
        self.adj_rec = adj_rec - np.diag(adj_rec)
        # percent_val = np.percentile(adj_rec, self.percentile)
        # adj_rec = np.where(adj_rec >= percent_val, np.ones_like(adj_rec), np.zeros_like(adj_rec))
        # #adj_rec = np.round(adj_rec)
        
        # #adj_re = np.reshape(construction, (feas['num_nodes'], feas['num_nodes']))
        # adj_rec = sp.csr_matrix(adj_rec)

        # adj_orig = adj_rec.copy()
        # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        # adj_orig.eliminate_zeros() 
        # self.adj_rec = adj_orig
    def sparse(self,percentile):
        percent_value = np.percentile(self.adj_rec, percentile)
        adj = self.adj_rec.copy()
        adj = np.where(adj >= percent_value, np.ones_like(adj), np.zeros_like(adj))
        adj = sp.csr_matrix(adj,dtype='int32')
        adj.eliminate_zeros() 
        return adj




# if __name__ == '__main__':
#     data_name = 'citeseer'
#     data = format_data(data_name)
#     adj = data['adj_orig']
#     features = data['features_orig']
#     model = 'arga_vae'
#     iterations = 50
#     runner = Link_rec_Runner(adj, features, model, iterations)  
#     runner.erun()


