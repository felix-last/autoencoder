import numpy as np
from theano import *
import theano.tensor as T

import pandas as pd
import matplotlib.pyplot as plt

import sklearn.utils

class Autoencoder:
    def __init__(self, layers, training_set, activation_fn=T.nnet.sigmoid):
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

        eta = self.eta = theano.shared(0.0,name='eta')

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

        self.MSSE = self._euclidian_distance(X,Y).mean()

        self.stats = []
        self.cost = self.MSSE
        self.prepare_backprop()
        self.clustering = False

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
        
    def prepare_backprop(self):#, x, eta):
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
                first_term = T.grad(self.cost, wrt=A[-1]).mean(axis=0)
            else:
                first_term = ( W[-(layer-1)].dot(error_l) )

            error_l = first_term * grad_a_wrt_z.mean(axis=0)

            W_n = self.W[-layer]
            b_n = self.b[-layer]

            A_prime = A[-(layer+1)].sum(axis=0)
            A_prime = A_prime.reshape((A_prime.shape[0], 1)) # shape to (k x 1) so that matrix multiplication is possible
            
            delta_W_n = -eta * (A_prime * error_l)
            delta_b_n = -eta * (error_l)
            
            updates_list.append((W_n, W_n + delta_W_n))
            updates_list.append((b_n, b_n + delta_b_n))
        self.update_weights_and_biases = function([self.X], updates=updates_list)

    def collect_stats(self, epoch):
        if self.training_set is not None:
            current = {
                'epoch': epoch,
                'Total Cost': np.asscalar(function([self.X],self.cost)(self.training_set)),
                'eta': self.eta.get_value()
            }
            if self.clustering:
                current['MSSE'] = np.asscalar(function([self.X],self.MSSE)(self.training_set))
                current['Clustering Cost'] = np.asscalar(function([self.X],self.clustering_cost)(self.training_set))
                current['Clustering Cost * q'] = np.asscalar(function([self.X],self.q*self.clustering_cost)(self.training_set))
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

    def adjust_eta(self, eta_strategy, epoch):
        self.eta.set_value(
                eta_strategy(self.eta.get_value(), epoch)
            )
    def train(self, epochs, learning_rate, minibatch_size, eta_strategy=(lambda eta, epoch: eta), collect_stats_every_nth_epoch=1):
        self.cost = self.MSSE
        eta = self.eta
        eta.set_value(learning_rate)
        next_epoch = self.get_stats()[-1]['epoch'] + 1
        if minibatch_size < 1:
            minibatch_size = minibatch_size * self.training_set.shape[0]
        self.collect_stats(0)
        for epoch in range(1,epochs+1):
            for batch_no in range(0,self.training_set.shape[0] // minibatch_size):
                minibatch = sklearn.utils.shuffle(self.training_set, n_samples=minibatch_size)
                self.update_weights_and_biases(minibatch)
            if (epoch % collect_stats_every_nth_epoch) == 0:
                self.collect_stats(epoch)
            self.adjust_eta(eta_strategy, epoch)

    def prepare_clustering(self, data, k, q):
        self.clustering = True
        X = self.X
        A = self.A
        
        H = self.H = A[self.layer_count // 2] # middle hidden layer, mapping function

        self.initialize_clusters_and_assignment(data, k, q)
        C = self.C # cluster centroids on H
        
        # C_closest_index = self.C_closest_index = T.argmin(((C-H)**2).sum(axis=1)) # index of closest cluster centroid to a given observation
        # C_closest = self.C_closest = C[C_closest_index] # closest cluster centroid to a given observation
        self.cluster_assignment, _ = theano.scan(
            fn=lambda h, C: T.argmin( self._euclidian_distance(h,C) ),#.astype('int32'),
            sequences=H,
            non_sequences=C
        )
        self.distance_closest_cluster, _ = theano.scan(
            fn=lambda h, C: T.min( self._euclidian_distance(h,C) ),
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
            return theano.ifelse.ifelse(
                T.neq(assigned_observations.size, 0), # if there are assigned_observations
                assigned_observations.mean(axis=0), # then return their mean,
                H[k] # else use the k-th observation as the centroid (assuming data is shuffled)
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
        self.current_cluster_assignment.set_value( function([self.X], self.cluster_assignment)(data))
    def update_cluster_centroids(self,data):
        self.C.set_value( function([self.X], self.cluster_centroids)(data) )

    def cluster(self, epochs, learning_rate, q, k, minibatch_size, q_msse_threshold=-1, eta_strategy=(lambda eta, epoch: eta), collect_stats_every_nth_epoch=1, plot_clusters_every_nth_epoch=1):
        self.eta.set_value(learning_rate)
        if minibatch_size < 1:
            minibatch_size = minibatch_size * self.training_set.shape[0]

        clustering_sample = sklearn.utils.shuffle( # use a random sample of ..% for clustering  
            self.training_set, 
            n_samples=minibatch_size#(self.training_set.shape[0]//4)
        )
        self.prepare_clustering(data=clustering_sample, k=k, q=q)
        if q_msse_threshold > 0:
            self.q.set_value(0)
        self.cost = self.MSSE + (self.q * self.clustering_cost)    

        self.collect_stats(0)
        for epoch in range(1, epochs+1):
            for batch_no in range(0,self.training_set.shape[0] // minibatch_size):
                minibatch = sklearn.utils.shuffle(self.training_set, n_samples=minibatch_size) # use a random sample of 10% for clustering
                self.update_weights_and_biases(minibatch)
            if (epoch % collect_stats_every_nth_epoch) == 0:
                self.collect_stats(epoch)
            self.adjust_eta(eta_strategy, epoch)
            if(self.q.get_value() != 0):
                self.update_cluster_centroids(clustering_sample)
                self.update_cluster_assignment(clustering_sample)
                if epoch == 1 or (epoch % plot_clusters_every_nth_epoch) == 0:
                    self.plot_clusters(clustering_sample, epoch)          

            if q_msse_threshold > 0 and self.q.get_value() == 0:
                if epoch % 10 == 0:
                    msse = function([self.X],self.MSSE)(self.training_set)
                    print(epoch, msse)
                    if msse < q_msse_threshold:
                        print('Epoch', epoch, ': MSSE', msse, 'is now smaller than Q-MSSE-Threshold', q_msse_threshold)
                        print('Beginning clustering.')
                        self.q.set_value(q)
                        self.update_cluster_assignment(clustering_sample)
                        self.update_cluster_centroids(clustering_sample)
        self.update_cluster_assignment(self.training_set)
        self.update_cluster_centroids(self.training_set)
        self.plot_clusters(self.training_set, 'Final (entire set)')


    def plot_clusters(self, data, epoch):
        np_H = theano.function([self.X],self.H)(data)
        np_assignment = self.current_cluster_assignment.get_value()#function([self.X], self.cluster_assignment)(data)
        np_cluster_centroids = self.C.get_value()#function([self.X], self.cluster_centroids)(data)
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

    def _euclidian_distance(self, a, b):
        return ((a-b)**2).sum(axis=1)

