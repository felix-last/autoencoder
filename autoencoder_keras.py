from keras.layers import Input, Dense
from keras.models import Model
import keras.optimizers
from keras import regularizers

class Autoencoder:
	def __init__(self, layers, training_set, regularize=False, optimizer='adadelta', loss='binary_crossentropy', validation_set=None):
		self.training_set = training_set
		self.validation_set = validation_set
		h_index = len(layers) // 2
		encoded_dim = layers[h_index]

		# Set up the layer structure
		X = Input(shape=(layers[0],))
		self.A = [X]
		for layer, units in enumerate(layers):
			if layer is 0: continue
			activation_fn = 'sigmoid' if layer == len(layers)-1 else 'relu'
			if regularize:
				activity_regularizer = regularizers.activity_l2()
				# W_regularizer = regularizers.l2()
			else:
				activity_regularizer =  None
				W_regularizer = None
			self.A.append(
				Dense(
					units, 
					activation=activation_fn, 
					activity_regularizer=activity_regularizer, 
					# TODO: Check that we're using a good (uniform?) initialization
					# W_regularizer=W_regularizer
				)(self.A[layer-1])
			)

		self.model = Model(input=self.A[0], output=self.A[-1])
		self.model.compile(optimizer=optimizer, loss=loss)
		
		self.encoder = Model(input=self.A[0], output=self.A[h_index])
		encoded_input = Input(shape=(encoded_dim,))

		decoder_layer_chain = self.model.layers[-h_index](encoded_input)
		for i in range(h_index-1, 0, -1):
			decoder_layer_chain = self.model.layers[-i](decoder_layer_chain)
		self.decoder = Model(input=encoded_input, output=decoder_layer_chain)
		
		self.encode = lambda data: self.encoder.predict(data)
		self.decode = lambda data: self.decoder.predict(data)
		self.fit = self.model.fit
		self.get_weights = self.get_parameters = self.model.get_weights
		self.set_weights = self.set_parameters = self.model.set_weights
		#self.decoder = Model(input=encoded_input, output=self.A[-1])





	#################################
	###########  LEGACY  ############
	#################################
	def train(self, epochs, eta, minibatch_size, mu=0.0, collect_stats_every_nth_epoch=1, verbose=False):
		#sgd = keras.optimizers.SGD(lr=eta, momentum=mu, nesterov=False)
		self.model.compile(optimizer='adadelta', loss='mse')
		self.model.fit(self.training_set, self.training_set,
                nb_epoch=epochs,
                batch_size=minibatch_size,
                validation_data=(self.validation_set, self.validation_set),
                shuffle=True,
                verbose= 1 if verbose else 0
        )

	def get_cost(self, dataset):
		self.model.compile(optimizer='sgd', loss='mse')
		return self.model.evaluate(dataset,dataset, verbose=0)
