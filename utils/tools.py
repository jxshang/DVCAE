import copy
import math
import random
import dgl
import torch
import datetime

import numpy as np
from torch.utils.data import Dataset
from dgl.data import DGLDataset




class dglDataset(DGLDataset):

    def __init__(self, input_vae, input_global, label, graph_nx, b_size, max_length, is_train=True):
        super(dglDataset, self).__init__(name='casdgl')
        self.vae, self.input_global, self.y, self.graph_nx = input_vae, input_global, label, graph_nx
        self.batch_size = b_size
        self.max_cascade_length = max_length
        self.is_train = is_train

        # graph
        self.graphs_dgl = list()
        for index, graph in enumerate(self.graph_nx):
            graph = dgl.DGLGraph(graph)
            x = np.concatenate([np.array(self.vae[index]), np.array(self.input_global[index])], axis=1)
            graph.ndata['x'] = torch.tensor(x).to(torch.float32)
            self.graphs_dgl.append(graph)

        # x
        for vae in self.vae:
            while len(vae) < self.max_cascade_length:
                vae.append(np.zeros(shape=len(vae[0])))
        for glo in self.input_global:
            while len(glo) < self.max_cascade_length:
                glo.append(np.zeros(shape=len(glo[0])))

        self.x = np.concatenate([np.array(self.vae), np.array(self.input_global)], axis=2)



    def process(self):
        pass

    def __getitem__(self, index):
        g = self.graphs_dgl[index]
        y = self.y[index]
        x = self.x[index]
        sample = {'x':x, 'y':y, 'g':g}
        return sample


    def __len__(self):
        return len(self.y)



class EarlyStopping(object):
    def __init__(self, patience, dataset):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_MSLE = None
        self.best_MAPE = None
        self.best_loss = None
        self.early_stop = False
        self.dataset = dataset

    def step(self, loss, MSLE, MAPE, model):
        if self.best_loss is None:
            self.best_MSLE = MSLE
            self.best_MAPE = MAPE
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (MSLE > self.best_MSLE):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (MSLE <= self.best_MSLE):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_MSLE = np.min((MSLE, self.best_MSLE))
            self.best_MAPE = np.min((MAPE, self.best_MAPE))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(),   './results/'+ self.dataset +'/{}'.format(self.filename))

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load('./results/'+ self.dataset +'/{}'.format(self.filename)))



"""

class Sampling2D(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(.5 * z_log_var) * epsilon
"""

"""

class Sampling3D(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        seq = tf.shape(z_mean)[1]
        dim = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, seq, dim))

        return z_mean + tf.exp(.5 * z_log_var) * epsilon
"""


"""
def nf_transformations(z, dim, k):
    z0 = z
    logD_loss = 0

    zk, logD = PlanarFlowLayer(dim, True)(z0)

    for i in range(k):
        zk, logD = PlanarFlowLayer(dim, False)((zk, logD))
        logD_loss += logD

    return zk, logD_loss
"""

"""
class PlanarFlowLayer(tf.keras.layers.Layer):
    def __init__(self,
                 z_dim,
                 is_first_layer=True):
        super(PlanarFlowLayer, self).__init__()
        self.z_dim = z_dim
        self.is_first_layer = is_first_layer

        self.w = self.add_weight(shape=(1, self.z_dim,), initializer='random_normal', trainable=True)
        self.u = self.add_weight(shape=(1, self.z_dim,), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)

    def call(self, inputs, **kwargs):
        EPSILON = 1e-7

        if self.is_first_layer:
            z_prev = inputs
        else:
            z_prev, sum_log_det_jacob = inputs
        m = lambda x: -1 + tf.math.log(1 + tf.exp(x))
        h = lambda x: tf.tanh(x)
        h_prime = lambda x: 1 - h(x) ** 2
        u_hat = (m(tf.tensordot(self.w, self.u, 2)) - tf.tensordot(self.w, self.u, 2)) \
                * (self.w / tf.norm(self.w)) + self.u
        z_prev = z_prev + u_hat * h(tf.expand_dims(tf.reduce_sum(z_prev * self.w, -1), -1) + self.b)
        affine = h_prime(tf.expand_dims(tf.reduce_sum(z_prev * self.w, -1), -1) + self.b) *self.w
        if self.is_first_layer:
            sum_log_det_jacob = tf.math.log(EPSILON + tf.abs(1 + tf.reduce_sum(affine * u_hat, -1)))
        else:
            sum_log_det_jacob += tf.math.log(EPSILON + tf.abs(1 + tf.reduce_sum(affine * u_hat, -1)))

        return z_prev, sum_log_det_jacob
"""