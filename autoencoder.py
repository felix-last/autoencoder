import numpy as np
from theano import *
import theano.tensor as T

import pandas as pd
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self, layers=[3,2,3], training_set=None, activation_fn=T.nnet.sigmoid):
        self.layers = layers
        self.layer_count = len(layers)
        self.features_count = layers[0]
        x = self.x = T.dvector('x')
        w = self.w = []
        b = self.b = []
        z = self.z = []
        a = self.a = [x]
        eta = self.eta = theano.shared(0.1,name='eta')

        for layer, units in enumerate(layers):
            if layer is 0: continue
            w_n = self.initialize_weights(rows=layers[layer-1], columns=units, number=layer)
            b_n = self.initialize_biases(units, layer)
            z_n = a[layer-1].dot(w_n) + b_n
            a_n = activation_fn(z_n)
            w.append(w_n)
            b.append(b_n)
            z.append(z_n)
            a.append(a_n)

        y = self.y = a[-1]    
        self.cost_graph = ((y - x)**2).sum()
        self.compute_cost = function([self.x],self.cost_graph)


        self.training_set = training_set
        self.stats = []
        self.collect_stats(0,0)
        self.prepare_backprop()

    def feedforward(self, x):
        return function([self.x], self.a[-1], name="feedforward")(x)

    def encode(self, x):
        middle_layer = (self.layer_count/2).floor()
        return function([self.x], self.a[middle_layer], name="encode")(x)

    def initialize_weights(self, rows, columns, number, method='random'):
        return theano.shared(
            numpy.random.randn(rows, columns), name="w{0}".format(number)
        )
    def initialize_biases(self, rows, number, method='random'):
        return theano.shared(
            numpy.random.randn(rows), name="b{0}".format(number)
        )
        
    def prepare_backprop(self):#, x, eta):
        cost_graph = self.cost_graph
        a = self.a
        z = self.z
        w = self.w
        b = self.b
        eta = self.eta
        error_L = T.grad(cost_graph, wrt=a[-1]) * T.jacobian(a[-1], wrt=z[-1]).diagonal()
        gradients_wrt_w = []
        gradients_wrt_b = []
        updates_list = []
        for layer in range(1, self.layer_count):
            if layer is 1:
                error_l = error_L
            else:
                error_l = ( w[-(layer-1)].dot(previous_error) ) * T.jacobian(a[-layer], wrt=z[-layer]).diagonal()
            previous_error = error_l
            a_n_prime = a[-(layer+1)] 
            a_n_prime = a_n_prime.reshape((a_n_prime.shape[0], 1)) # shape to (k x 1) so that matrix multiplication is possible
            delta_w_n = -eta * (a_n_prime * error_l)             
            delta_b_n = -eta * (error_l)
            w_n = self.w[-layer]
            b_n = self.b[-layer]
            updates_list.append((w_n, w_n + delta_w_n))
            updates_list.append((b_n, b_n + delta_b_n))
        self.update_weights_and_biases = function([self.x], updates=updates_list)


    def backprop_SGD(self, x):
        self.update_weights_and_biases(x)
        
    def compute_avg_cost(self, dataset):
        avg_cost = 0
        for r in dataset:
            avg_cost += self.compute_cost(r) / dataset.shape[0]
        return avg_cost

    def collect_stats(self, epoch, eta):
        if self.training_set is not None:
            self.stats.append({
                    'epoch': epoch,
                    'average_cost': self.compute_avg_cost(self.training_set),
                    'eta': eta
                })

    def get_stats(self, as_df=False):
        if as_df:
            df_stats = pd.DataFrame(self.stats)
            if len(self.stats) > 0:
                df_stats.set_index('epoch')
                df_stats.index.set_names(['epoch'], inplace=True)
            return df_stats
        else:
            return self.stats

    def plot_stats(self):
        self.get_stats(as_df=True)[['average_cost']].plot()

    def train(self, epochs, learning_rate, eta_strategy=(lambda eta, epoch: eta), collect_stats=True):
        eta = self.eta
        eta.set_value(learning_rate)
        next_epoch = self.get_stats()[-1]['epoch'] + 1
        for epoch in range(next_epoch,next_epoch + epochs):
            for row in self.training_set:
                self.backprop_SGD(row)
            if collect_stats:
                self.collect_stats(epoch, eta.get_value())
            eta.set_value(
                eta_strategy(eta.get_value(), epoch)
            )