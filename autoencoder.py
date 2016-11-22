import numpy as np
from theano import *
import theano.tensor as T

import pandas as pd
import matplotlib.pyplot as plt

import sklearn.utils

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
        self.current_cost = None
        self.prepare_backprop(self.cost_graph)

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
        
    def prepare_backprop(self, cost):#, x, eta):
        if self.current_cost is cost:
            return
        else:
            self.current_cost = cost

        a = self.a
        z = self.z
        w = self.w
        b = self.b
        eta = self.eta
        error_L = T.grad(cost, wrt=a[-1]) * T.jacobian(a[-1], wrt=z[-1]).diagonal()
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

    def train(self, epochs, learning_rate, cost, eta_strategy=(lambda eta, epoch: eta), collect_stats=True):
        self.prepare_backprop(cost=cost)
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

    def prepare_clustering(self, q, k, dataset):
        x = self.x
        a = self.a
        self.k = k
        
        h_dim = self.h_dim = self.layers[self.layer_count // 2] # number of units on mapping layer

        C = self.C = theano.shared(
            np.random.rand(k,h_dim),
            name='C'
        ) # matrix of cluster centroids

        if hasattr(self, 'q') and self.k is k and self.q is q:
            return
        
        h = self.h = a[self.layer_count // 2] # middle hidden layer, mapping function
        self.encode = function([x], h)
        
        c_closest_index = self.c_closest_index = T.argmin(((C-h)**2).sum(axis=1)) # index of closest cluster centroid to a given observation
        c_closest = self.c_closest = C[c_closest_index] # closest cluster centroid to a given observation
        
        self.q = theano.shared(q, name='q') # clustering hyper parameter
        
        self.clustering_cost_graph = self.cost_graph + self.q * ((h - c_closest)**2).sum()
        self.compute_cost_for_clustering = function([x], self.clustering_cost_graph)
        
        clustered_dataset = [ [] for cluster_no in range(0,k) ]
        for i, observation in enumerate(dataset):
            cluster_no = i % k # assumes an already shuffled set!!
            clustered_dataset[cluster_no].append(self.encode(observation))
        self.clustered_dataset = np.asarray( [ np.asarray(cluster) for cluster in clustered_dataset ] )


    def cluster(self, epochs, learning_rate, q, k, eta_strategy=(lambda eta, epoch: eta), collect_stats=True):
        sample_set = sklearn.utils.shuffle(self.training_set, n_samples=(self.training_set.shape[0]//10)) # use a random sample of 10% for clustering
        self.prepare_clustering(q,k,dataset=sample_set)
        self.prepare_backprop(cost=self.clustering_cost_graph)
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
            self.update_cluster_centroids(sample_set)
            self.assign_clusters(sample_set)
            #for c_i in self.C:
            #    print(c_i)
        self.update_cluster_centroids(self.training_set)
        self.assign_clusters(self.training_set)

    def assign_clusters(self, dataset):
        x = self.x
        clustered_dataset = [ [] for cluster_no in range(0,self.k) ]
        for observation in dataset:
            cluster_no = function([x], outputs=self.c_closest_index)(observation)
            clustered_dataset[cluster_no].append(self.encode(observation))
        self.assigned_last_to = self.C
        self.clustered_dataset = np.asarray( [ np.asarray(cluster) for cluster in clustered_dataset ] )
        return self.clustered_dataset

    def update_cluster_centroids(self, dataset):
        clustered_dataset = self.clustered_dataset
        new_cluster_centroids = np.empty((self.k, self.h_dim))
        for cluster_no, observations in enumerate(clustered_dataset):
            new_cluster_centroids[cluster_no] = observations.mean(axis=0)
        self.C.set_value(new_cluster_centroids)

    def plot_clusters(self):
        plt.figure()
        colors = ['b','r','g','c','m','y','k','w']
        for i, cluster in enumerate(self.clustered_dataset):
            plt.scatter(
                x=cluster.transpose()[0],
                y=cluster.transpose()[1], 
                marker='x', 
                c=colors[i]
            )

        for i, centroid in enumerate(self.C.get_value()):
            plt.scatter(
                x=centroid.transpose()[0],
                y=centroid.transpose()[1], 
                marker='o', 
                c=colors[i]
            )

