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

    def __init__(self, layers, training_set, activation_fn=T.nnet.sigmoid, initialize_parameters_method='random'):
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
        prev_grad_W = self.prev_grad_W = []
        prev_grad_b = self.prev_grad_b = []

        eta = self.eta = theano.shared(0.0,name='eta')
        mu = self.mu = theano.shared(0.0, name='mu')

        for layer, units in enumerate(layers):
            if layer is 0: continue
            W_n = self.initialize_weights(rows=layers[layer-1], columns=units, number=layer, method=initialize_parameters_method)
            b_n = self.initialize_biases(units, layer, method=initialize_parameters_method)
            prev_grad_W_n = theano.shared(np.zeros((layers[layer-1], units)), 'prev_grad_W_n')
            prev_grad_b_n = theano.shared(np.zeros(units), 'prev_grad_b_n')
            Z_n = A[layer-1].dot(W_n) + b_n
            A_n = activation_fn(Z_n)

            W.append(W_n)
            b.append(b_n)
            Z.append(Z_n)
            A.append(A_n)
            prev_grad_W.append(prev_grad_W_n)
            prev_grad_b.append(prev_grad_b_n)

        H = self.H = A[self.layer_count // 2] # middle hidden layer, mapping function
        Y = self.Y = A[-1]
        self.encode = theano.function([X],H)
        self.decode = theano.function([H],Y)
        
        self.euclidian_distance = lambda a, b: ((a-b)**2).sum(axis=1)
        self.MSSE = self.euclidian_distance(X,Y).mean()

        self.stats = []
        self.cost = self.MSSE
        self.scaffold_backprop()
        self.clustering = False

    def initialize_weights(self, rows, columns, number, method='random'):
        if method is 'random':
            W_n = np.random.randn(rows, columns)
        elif method is 'ones':
            W_n = np.ones((rows, columns))
        return theano.shared(
            W_n, name="W{0}".format(number)
        )
    def initialize_biases(self, rows, number, method='random'):
        if method is 'random':
            b_n = np.random.randn(rows)
        elif method is 'ones':
            b_n = np.ones(rows)
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
        prev_grad_W = self.prev_grad_W
        prev_grad_b = self.prev_grad_b
        mu = self.mu
        
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
                first_term = T.grad(self.cost, wrt=A[-1]).mean(axis=0)
            else:
                first_term = ( W[-(layer-1)].dot(error_l) )

            error_l = first_term * grad_a_wrt_z.mean(axis=0)

            W_n = W[-layer]
            b_n = b[-layer]

            A_prime = A[-(layer+1)].sum(axis=0)
            A_prime = A_prime.reshape((A_prime.shape[0], 1)) # shape to (k x 1) so that matrix multiplication is possible
            
            delta_W_n = -eta * ( (A_prime * error_l) + (mu * prev_grad_W[-layer]) )
            delta_b_n = -eta * ( (error_l) + (mu * prev_grad_b[-layer]) )
            
            updates_list.append((W_n, W_n + delta_W_n))
            updates_list.append((b_n, b_n + delta_b_n))
            updates_list.append((prev_grad_W[-layer], delta_W_n))
            updates_list.append((prev_grad_b[-layer], delta_b_n))
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

    def perform_training_epoch(self, epoch):
        for batch_no in range(0,self.training_set.shape[0] // self.minibatch_size):
            minibatch = sklearn.utils.shuffle(self.training_set, n_samples=self.minibatch_size) # use a random sample of 10% for clustering
            self.update_weights_and_biases(minibatch)

        if (epoch % self.collect_stats_every_nth_epoch) == 0:
            self.collect_stats(epoch)
            self.eta.set_value(
                self.eta_strategy(self.eta.get_value(), epoch)
            )
    
    def train(self, epochs, eta, minibatch_size, mu=0.0, eta_strategy=None, collect_stats_every_nth_epoch=1):
        self.set_training_params(eta, mu, minibatch_size, eta_strategy, collect_stats_every_nth_epoch)
        
        self.collect_stats(0)
        start_time = time()
        for epoch in range(1, epochs+1):
            self.perform_training_epoch(epoch)
        time_elapsed = time() - start_time
        print('Total Time:', time_elapsed, 'Per Epoch ~', time_elapsed/epochs)






    ###############################################################
    ######################### CLUSTERING  #########################
    ###############################################################

    def scaffold_clustering(self, data, k, q):
        self.clustering = True
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
        self.clustering_cost = self.per_cluster_cost.sum()

        def calculate_cluster_centroid(k, H, assignment):
            assigned_observations = (H[T.eq(assignment, T.fill(assignment,k)).nonzero()])
            furthest_observation_from_its_centroid = H[T.argmin(self.distance_closest_cluster)]
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
        self.current_cluster_assignment.set_value( theano.function([self.X], self.cluster_assignment)(data))
    def update_cluster_centroids(self,data):
        self.C.set_value( theano.function([self.X], self.cluster_centroids)(data) )

    def cluster(self, epochs, eta, q, k, minibatch_size, mu=0.0, q_msse_threshold=-1, eta_strategy=None, collect_stats_every_nth_epoch=1, plot_clusters_every_nth_epoch=-1):
        self.set_training_params(eta, mu, minibatch_size, eta_strategy, collect_stats_every_nth_epoch)
        
        # Clustering Preparation
        clustering_sample = sklearn.utils.shuffle(self.training_set, n_samples=minibatch_size)
        self.scaffold_clustering(data=clustering_sample, k=k, q=q)
        if q_msse_threshold > 0:
            self.q.set_value(0)
        self.cost = self.MSSE + (self.q * self.clustering_cost)    

        self.collect_stats(0)
        start_time = time()
        for epoch in range(1, epochs+1):
            self.perform_training_epoch(epoch)

            if(self.q.get_value() > 0):
                self.update_cluster_centroids(clustering_sample)
                self.update_cluster_assignment(clustering_sample)
                if (epoch == 1 or (epoch % plot_clusters_every_nth_epoch) == 0) and plot_clusters_every_nth_epoch > 0:
                    self.plot_clusters(clustering_sample, epoch)          
            else:
                if epoch % 10 == 0:
                    if q_msse_threshold > 0:
                        msse = theano.function([self.X],self.MSSE)(self.training_set)
                        print(epoch, msse)
                        if msse < q_msse_threshold:
                            print('Epoch', epoch, ': MSSE', msse, 'is now smaller than Q-MSSE-Threshold', q_msse_threshold)
                            print('Beginning clustering.')
                            self.q.set_value(q)
                            self.update_cluster_assignment(clustering_sample)
                            self.update_cluster_centroids(clustering_sample)
        
        time_elapsed = time() - start_time
        print('Total Time:', time_elapsed, 'Per Epoch ~', time_elapsed/epochs)

        self.update_cluster_assignment(self.training_set)
        self.update_cluster_centroids(self.training_set)
        if plot_clusters_every_nth_epoch > 0:
            self.plot_clusters(self.training_set, 'Final (entire set)')





    ###############################################################
    #################### LOGGING AND PLOTTING #####################
    ###############################################################

    def collect_stats(self, epoch):
        if self.training_set is not None:
            current = {
                'epoch': epoch,
                'Total Cost': np.asscalar(theano.function([self.X],self.cost)(self.training_set)),
                'eta': self.eta.get_value()
            }
            if self.clustering:
                current['MSSE'] = np.asscalar(theano.function([self.X],self.MSSE)(self.training_set))
                current['Clustering Cost'] = np.asscalar(theano.function([self.X],self.clustering_cost)(self.training_set))
                current['Clustering Cost * q'] = np.asscalar(theano.function([self.X],self.q*self.clustering_cost)(self.training_set))
            self.stats.append(current)

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
            if stacked:
                stats[['MSSE', 'Clustering Cost * q']].plot.area(title=title)
            else:
                return stats.plot()
        else:
            print('Nothing to plot')
    def plot_clusters(self, data, epoch):
        np_H = theano.theano.function([self.X],self.H)(data)
        np_assignment = self.current_cluster_assignment.get_value()#theano.function([self.X], self.cluster_assignment)(data)
        np_cluster_centroids = self.C.get_value()#theano.function([self.X], self.cluster_centroids)(data)
        colors = ['b','r','g','c','m','y','k','w']
        if np_cluster_centroids.shape[0] > len(colors):
            colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f'] + colors
        plt.figure()
        #plt.axis([0,1,0,1])
        plt.axis('equal')
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