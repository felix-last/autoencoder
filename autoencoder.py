import numpy as np
import theano
import theano.tensor as T

import pandas as pd
import matplotlib.pyplot as plt

import sklearn.utils

from time import time

class Autoencoder:





    ###############################################################
    ################### Network Initialization ####################
    ###############################################################

    def __init__(self, layers, training_set, validation_set, activation_fn=T.nnet.sigmoid, initialize_parameters_method='random'):
        self.layers = layers
        self.layer_count = len(layers)
        self.features_count = layers[0]
        self.activation_fn = activation_fn
        self.training_set = training_set
        self.validation_set = validation_set
        X = self.X = T.dmatrix('X')
        W = self.W = []
        b = self.b = []
        Z = self.Z = []
        A = self.A = [X]
        prev_delta_W = self.prev_delta_W = []
        prev_delta_b = self.prev_delta_b = []

        eta = self.eta = theano.shared(0.0,name='eta')
        mu = self.mu = theano.shared(0.0, name='mu')

        for layer, units in enumerate(layers):
            if layer is 0: continue
            W_n = self.initialize_weights(rows=layers[layer-1], columns=units, number=layer, method=initialize_parameters_method)
            b_n = self.initialize_biases(units, layer, method=initialize_parameters_method)
            prev_delta_W_n = theano.shared(np.zeros((layers[layer-1], units)), 'prev_delta_W_n')
            prev_delta_b_n = theano.shared(np.zeros(units), 'prev_delta_b_n')
            Z_n = A[layer-1].dot(W_n) + b_n
            A_n = activation_fn(Z_n)

            W.append(W_n)
            b.append(b_n)
            Z.append(Z_n)
            A.append(A_n)
            prev_delta_W.append(prev_delta_W_n)
            prev_delta_b.append(prev_delta_b_n)

        H = self.H = A[self.layer_count // 2] # middle hidden layer, mapping function
        Y = self.Y = A[-1]
        self.encode = theano.function([X],H)
        self.decode = theano.function([H],Y)
        self.feedforward = theano.function([X],Y)
        
        self.euclidian_distance = lambda a, b: ((a-b)**2).sum(axis=1)
        # mean of sum of squares cost
        self.MSSE = self.euclidian_distance(X,Y).mean()
        # mean of mean squared errors cost
        self.MSE = ((X-Y)**2).mean()
        # cross-entropy cost
        self.CE = - T.mean( X * T.log(Y) + (T.ones_like(X)-X) * T.log(T.ones_like(X)-Y) )
        self.default_cost = self.CE

        self.stats = []
        self.cost = self.default_cost
        self.get_cost = theano.function([self.X], self.cost)
        self.clustering = False
        self.scaffold_backprop()

    def initialize_weights(self, rows, columns, number, method='random'):
        if method is 'random':
            W_n = np.random.randn(rows, columns)
        elif method is 'ones':
            W_n = np.ones((rows, columns))
        elif method is 'zeros':
            W_n = np.zeros((rows, columns))
        return theano.shared(
            W_n, name="W{0}".format(number)
        )
    def initialize_biases(self, rows, number, method='random'):
        if method is 'random':
            b_n = np.random.randn(rows)
        elif method is 'ones':
            b_n = np.ones(rows)
        elif method is 'zeros':
            b_n = np.zeros(rows)
        return theano.shared(
            b_n, name="b{0}".format(number)
        )





    ###############################################################
    ############### Training with SGD and Backprop ################
    ###############################################################

    def scaffold_backprop(self):#, x, eta):
        A = self.A
        Z = self.Z
        X = self.X
        W = self.W
        b = self.b
        eta = self.eta
        prev_delta_W = self.prev_delta_W
        prev_delta_b = self.prev_delta_b
        mu = self.mu
        
        updates_list = []

        for layer in range(1, self.layer_count):
            cost = self.cost
            if self.clustering and layer <= self.layer_count / 2: cost = self.default_cost # only encoding weights can be trained with clustering cost

            W_n = W[-layer]
            b_n = b[-layer]
            prev_delta_W_n = prev_delta_W[-layer]
            prev_delta_b_n = prev_delta_b[-layer]

            delta_W_n = -eta * T.grad(cost, wrt=W_n) + mu * prev_delta_W_n
            delta_b_n = -eta * T.grad(cost, wrt=b_n) + mu * prev_delta_b_n
            
            updates_list.append((W_n, W_n + delta_W_n))
            updates_list.append((b_n, b_n + delta_b_n))
            updates_list.append((prev_delta_W_n, delta_W_n))
            updates_list.append((prev_delta_b_n, delta_b_n))
        self.update_weights_and_biases = theano.function([X], updates=updates_list)
    
    def set_training_params(self, eta, mu, minibatch_size, eta_strategy, collect_stats_every_nth_epoch):
        self.eta.set_value(eta)
        self.mu.set_value(mu)
        self.collect_stats_every_nth_epoch = collect_stats_every_nth_epoch
        self.eta_strategy = eta_strategy or (lambda eta, epoch: eta)
        if minibatch_size < 1:
            self.minibatch_size = minibatch_size * self.training_set.shape[0]
        else:
            self.minibatch_size = minibatch_size
        self.stats = []
        self.last_validation_cost = self.get_cost(self.validation_set)
        self.last_validation_cost_age = 0

    def perform_training_epoch(self, epoch, verbose=None):
        shuffled_indices = sklearn.utils.shuffle(np.arange(0,len(self.training_set)))
        for batch_no in range(0,self.training_set.shape[0] // self.minibatch_size + 1):
            start_index = batch_no * self.minibatch_size
            end_index = start_index + self.minibatch_size
            if end_index >= len(self.training_set):
                end_index = len(self.training_set) - 1
            minibatch_indices = shuffled_indices[start_index:end_index]
            minibatch =  self.training_set[minibatch_indices]
            self.update_weights_and_biases(minibatch)

        self.validation_performance_is_decreasing = self.is_validation_performance_decreasing()

        if (epoch % self.collect_stats_every_nth_epoch) == 0:
            self.collect_stats(epoch, verbose)
            self.eta.set_value(
                self.eta_strategy(self.eta.get_value(), epoch)
            )
    
    def train(self, epochs, eta, minibatch_size, mu=0.0, eta_strategy=None, collect_stats_every_nth_epoch=1, verbose=None):
        self.set_training_params(eta, mu, minibatch_size, eta_strategy, collect_stats_every_nth_epoch)
        self.cost = self.default_cost
        
        self.collect_stats(0)        
        print('Starting training with initial training cost:', self.get_cost(self.training_set), 'and validation cost:', self.get_cost(self.validation_set))
        start_time = time()
        for epoch in range(1, epochs+1):
            self.perform_training_epoch(epoch, verbose)
            if self.validation_performance_is_decreasing: break

        time_elapsed = time() - start_time
        print('Training Time:', time_elapsed, 'Per Epoch ~', time_elapsed/epochs)

    def is_validation_performance_decreasing(self):
        self.current_validation_cost = self.get_cost(self.validation_set)
        if self.current_validation_cost >= self.last_validation_cost || np.isclose(self.current_validation_cost, self.last_validation_cost):
            self.last_validation_cost_age += 1
        else:
            self.last_validation_cost_age = 0
            self.last_validation_cost = self.current_validation_cost
        if self.last_validation_cost_age >= 15:
            print('Validation error has been unchanged or decreasing for', self.last_validation_cost_age, 'epochs. Stopping training.')
            return True
        else: return False






    ###############################################################
    ######################### CLUSTERING  #########################
    ###############################################################

    def scaffold_clustering(self, data, k, q):
        X = self.X
        A = self.A
        H = self.H

        self.initialize_clusters_and_assignment(data, k, q)
        C = self.C # cluster centroids on H
        
        self.cluster_assignment, _ = theano.scan(
            fn=lambda h, C: T.argmin( self.euclidian_distance(h,C) ),#.astype('int32'),
            sequences=H,
            non_sequences=C
        )
        self.distance_closest_cluster, _ = theano.scan(
            fn=lambda h, C: T.min( self.euclidian_distance(h,C) ),
            sequences=H,
            non_sequences=C
        )
        self.per_cluster_cost, _ = theano.scan(
                fn=lambda k, distances, assignment: (distances[T.eq(assignment, T.fill(assignment,k)).nonzero()]).sum(),
                sequences=[T.arange(C.shape[0])],
                non_sequences=[self.distance_closest_cluster,self.current_cluster_assignment]
            )
        #self.clustering_cost = self.per_cluster_cost.sum()
        self.clustering_cost = self.distance_closest_cluster.mean()

        def calculate_cluster_centroid(k, H, assignment):
            assigned_observations = (H[T.eq(assignment, T.fill(assignment,k)).nonzero()])
            furthest_observation_from_its_centroid = H[T.argmax(self.distance_closest_cluster)]
            return theano.ifelse.ifelse(
                T.neq(assigned_observations.size, 0), # if there are assigned_observations
                assigned_observations.mean(axis=0), # then return their mean,
                furthest_observation_from_its_centroid # else set the centroid to the furthest outside point
            )
        self.cluster_centroids, _ = theano.scan(
            fn=calculate_cluster_centroid,#lambda k, H, assignment: (H[T.eq(assignment, T.fill(assignment,k)).nonzero()]).mean(axis=0),
            sequences=[T.arange(C.shape[0])],
            non_sequences=[H,self.current_cluster_assignment]
        )
        
        self.update_cluster_centroids(data)
        self.update_cluster_assignment(data)

    def initialize_clusters_and_assignment(self, data, k, q):
        h_dim = self.layers[self.layer_count // 2] # number of units on mapping layer
        self.k = k
        self.q = theano.shared(q, name='q') # clustering hyper parameter
        self.C = theano.shared(
            np.random.rand(k,h_dim),
            name='C'
        )
        self.current_cluster_assignment = theano.shared((np.random.choice(k, data.shape[0])).astype('int64'), 'current_cluster_assignment')

    def update_cluster_assignment(self,data):
        self.current_cluster_assignment.set_value(self.get_cluster_assigment(data) )
    def update_cluster_centroids(self,data):
        self.C.set_value( theano.function([self.X], self.cluster_centroids)(data) )
    def get_cluster_assigment(self,data):
        return theano.function([self.X], self.cluster_assignment)(data)

    def cluster(self, epochs, eta, q, k, minibatch_size, mu=0.0, eta_strategy=None, collect_stats_every_nth_epoch=1, plot_clusters_every_nth_epoch=-1, verbose=None):
        self.set_training_params(eta, mu, minibatch_size, eta_strategy, collect_stats_every_nth_epoch)
        
        # Clustering Preparation
        clustering_sample = self.training_set #sklearn.utils.shuffle(self.training_set, n_samples=minibatch_size)
        self.scaffold_clustering(data=clustering_sample, k=k, q=q)
        self.cost = self.default_cost + (self.q * self.clustering_cost) 
        self.clustering = True
        self.scaffold_backprop()

        self.collect_stats(0)
        print('Starting training with initial training cost:', self.get_cost(self.training_set), 'and validation cost:', self.get_cost(self.validation_set))
        start_time = time()
        for epoch in range(1, epochs+1):
            self.perform_training_epoch(epoch, verbose)
            if self.validation_performance_is_decreasing: break

            self.update_cluster_centroids(clustering_sample)
            self.update_cluster_assignment(clustering_sample)
            if (epoch == 1 or (epoch % plot_clusters_every_nth_epoch) == 0) and plot_clusters_every_nth_epoch > 0:
                self.plot_clusters(clustering_sample, epoch)
        
        time_elapsed = time() - start_time
        print('Training Time:', time_elapsed, 'Per Epoch ~', time_elapsed/epochs)

        self.update_cluster_assignment(self.training_set)
        self.update_cluster_centroids(self.training_set)
        if plot_clusters_every_nth_epoch > 0:
            self.plot_clusters(self.training_set, 'Final (entire set)')





    ###############################################################
    #################### LOGGING AND PLOTTING #####################
    ###############################################################

    def collect_stats(self, epoch, verbose=False):
        current = {
            'epoch': epoch,
            'eta': np.asscalar(self.eta.get_value()),
            'Cost Validation': np.asscalar(self.last_validation_cost) # rely on this to be updated every epoch
        }
        if self.training_set is not None:
            theano_computations = [self.cost]
            if self.clustering:
                theano_computations.append(self.clustering_cost)
                theano_computations.append(self.clustering_cost * self.q)
            theano_compute = theano.function([self.X],theano_computations)
            current['Cost'], *other_results = (np.asscalar(arr) for arr in theano_compute(self.training_set))
            if len(other_results) > 0:
                current['Clustering Cost'], current['Clustering Cost * q'] = other_results
        self.stats.append(current)
        if verbose:
            print('Epoch', epoch, 'Cost:', current['Cost'], 'Cost Validation:', current['Cost Validation'])

    def get_stats(self, as_df=False):
        if as_df:
            df_stats = pd.DataFrame(self.stats)
            if len(self.stats) > 0:
                df_stats.set_index('epoch',inplace=True)
            return df_stats
        else:
            return self.stats

    def plot_stats(self, stacked=False, title=None):
        stats = self.get_stats(as_df=True)
        if len(stats) > 1:
            graphs = ['Cost', 'Cost Validation']
            if stacked:
                if self.clustering:
                    graphs.append('Clustering Cost * q')
                stats[graphs].plot.area(title=title)
            else:
                return stats[graphs].plot()
        else:
            print('Nothing to plot')
    def plot_clusters(self, data, epoch='', fixed_view=True):
        np_H = theano.theano.function([self.X],self.H)(data)
        np_assignment = self.current_cluster_assignment.get_value()#theano.function([self.X], self.cluster_assignment)(data)
        np_cluster_centroids = self.C.get_value()#theano.function([self.X], self.cluster_centroids)(data)
        colors = ['b','r','g','c','m','y','k','w']
        if np_cluster_centroids.shape[0] > len(colors):
            colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f'] + colors
        plt.figure()
        plt.axis([0,1,0,1]) if fixed_view else plt.axis('equal')
        if np_H.shape[1] < 2:
            y = np.ones_like(np_H)
        else:
            y = np_H[:,1]
        plt.scatter(
            x=np_H[:,0], 
            y=y, 
            marker='+', 
            color=[ colors[i] for i in np_assignment ]
        )
        if np_cluster_centroids.shape[1] < 2:
            y = np.ones_like(np_cluster_centroids)
        else:
            y = np_cluster_centroids[:,1]
        plt.scatter(
            x=np_cluster_centroids[:,0],
            y=y,
            marker='o',
            color=colors
        )
        plt.title('Epoch:' + str(epoch))
        plt.show()