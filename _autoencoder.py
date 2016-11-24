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
        self.activation_fn = activation_fn
        self.training_set = training_set
        X = self.X = T.dmatrix('X')
        W = self.W = []
        b = self.b = []
        Z = self.Z = []
        A = self.A = [X]

        eta = self.eta = theano.shared(0.1,name='eta')

        for layer, units in enumerate(layers):
            if layer is 0: continue
            W_n = self.initialize_weights(rows=layers[layer-1], columns=units, number=layer)
            b_n = self.initialize_biases(units, layer)
            Z_n = A[layer-1].dot(W_n) + b_n
            A_n = activation_fn(Z_n)

            W.append(W_n)
            b.append(b_n)
            Z.append(Z_n)
            A.append(A_n)

        Y = self.Y = A[-1]

        self.cost_graph = ((Y - X)**2).sum()
        self.compute_cost = function([X],self.cost_graph)
        print(self.compute_cost(self.training_set))

        
        self.stats = []
        #self.collect_stats(0,0)
        self.current_cost = None
        self.prepare_backprop(self.cost_graph)

    def feedforward(self, batch):
        return function([self.X], self.A[-1], name="feedforward")(batch)

    def initialize_weights(self, rows, columns, number, method='random'):
        return theano.shared(
            numpy.random.randn(rows, columns), name="W{0}".format(number)
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

        A = self.A
        Z = self.Z
        W = self.W
        b = self.b
        eta = self.eta
        
        gradients_wrt_w = []
        gradients_wrt_b = []
        updates_list = []
        
        def fn_grad_a_wrt_z(Z):
                A = self.activation_fn(Z) # have to redefine A here because theano doesn't recognize the subtensor relationship
                return theano.tensor.jacobian(A, wrt=Z).diagonal()

        for layer in range(1, self.layer_count):
            grad_a_wrt_z, _ = theano.scan(
                    fn=fn_grad_a_wrt_z, 
                    sequences=[Z[-layer]]
                )
            
            if layer is 1:
                error_l = T.grad(cost, wrt=A[-1]) * grad_a_wrt_z
                print(function([self.X], T.grad(cost, wrt=A[-1]))(self.training_set).shape)
                print(function([self.X], grad_a_wrt_z)(self.training_set).shape)
                print(function([self.X], error_l)(self.training_set).shape)
                
            else:
                error_l = ( W[-(layer-1)].dot(previous_error.transpose()) ).transpose() * grad_a_wrt_z
                print(layer, 'w.dot(e-1)',function([self.X], ( W[-(layer-1)].dot(previous_error.transpose()) ).transpose())(self.training_set).shape)
                print(layer, 'grad_a_wrt_z',function([self.X], grad_a_wrt_z)(self.training_set).shape)
                print(layer, 'error_l', function([self.X], error_l)(self.training_set).shape)

            previous_error = error_l
            A_n_prime = A[-(layer+1)] 
            print(layer, 'A_n_prime', function([self.X], A_n_prime)(self.training_set).shape)
            print(layer, 'error_l dot A_n_prime', function([self.X], error_l.transpose().dot(A_n_prime).transpose() )(self.training_set).shape)

            #A_n_prime = A_n_prime.transpose()#reshape((A_n_prime.shape[0], 1)) # shape to (k x 1) so that matrix multiplication is possible
            deltA_W_n = -eta * error_l.transpose().dot(A_n_prime).transpose() #(A_n_prime * error_l)             
            deltA_b_n = -eta * (error_l)
            W_n = self.W[-layer]
            b_n = self.b[-layer]
            
            print(layer, 'W_n', W_n.get_value().shape)
            print(layer, 'b_n', b_n.get_value().shape)

            updates_list.append((W_n, W_n + deltA_W_n))
            updates_list.append((b_n, b_n + deltA_b_n))
        self.update_weights_and_biases = function([self.X], updates=updates_list)


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

    def train(self, epochs, learning_rate, cost, etA_strategy=(lambda eta, epoch: eta), collect_stats=True):
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
                etA_strategy(eta.get_value(), epoch)
            )

    def prepare_clustering(self, q, k, dataset):
        X = self.X
        a = self.a
        self.k = k
        
        h_dim = self.h_dim = self.layers[self.layer_count // 2] # number of units on mapping layer

        C = self.C = theano.shared(
            np.random.rand(k,h_dim),
            name='C'
        ) # matrix of cluster centroids

        if hasattr(self, 'q') and self.k is k and self.q is q:
            return
        
        h = self.h = A[self.layer_count // 2] # middle hidden layer, mapping function
        self.encode = function([X], h)
        
        c_closest_index = self.c_closest_index = T.argmin(((C-h)**2).sum(axis=1)) # index of closest cluster centroid to a given observation
        c_closest = self.c_closest = C[c_closest_index] # closest cluster centroid to a given observation
        
        self.q = theano.shared(q, name='q') # clustering hyper parameter
        
        self.clustering_cost_graph = self.cost_graph + self.q * ((h - c_closest)**2).sum()
        self.compute_cost_for_clustering = function([X], self.clustering_cost_graph)
        
        clustered_dataset = [ [] for cluster_no in range(0,k) ]
        for i, observation in enumerate(dataset):
            cluster_no = i % k # assumes an already shuffled set!!
            clustered_dataset[cluster_no].append(self.encode(observation))
        self.clustered_dataset = np.asarray( [ np.asarray(cluster) for cluster in clustered_dataset ] )


    def cluster(self, epochs, learning_rate, q, k, etA_strategy=(lambda eta, epoch: eta), collect_stats=True):
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
                etA_strategy(eta.get_value(), epoch)
            )
            self.update_cluster_centroids(sample_set)
            self.assign_clusters(sample_set)
            #for c_i in self.C:
            #    print(c_i)
        self.update_cluster_centroids(self.training_set)
        self.assign_clusters(self.training_set)

    def assign_clusters(self, dataset):
        X = self.X
        
        
        clustered_dataset = [ [] for cluster_no in range(0,self.k) ]
        for observation in dataset:
            cluster_no = function([X], outputs=self.c_closest_index)(observation)
            clustered_dataset[cluster_no].append(self.encode(observation))
        self.assigned_last_to = self.C
        self.clustered_dataset = np.asarray( [ np.asarray(cluster) for cluster in clustered_dataset ] )
        return self.clustered_dataset

    def update_cluster_centroids(self, dataset):
        clustered_dataset = self.clustered_dataset
        neW_cluster_centroids = np.empty((self.k, self.h_dim))
        for cluster_no, observations in enumerate(clustered_dataset):
            neW_cluster_centroids[cluster_no] = observations.mean(axis=0)
        self.C.set_value(neW_cluster_centroids)

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

