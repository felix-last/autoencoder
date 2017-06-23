from keras.layers import Input, Dense
from keras.models import Model
import keras.optimizers
from keras import regularizers

class Autoencoder:
	def __init__(self, layers, training_set, regularize=False, optimizer='adadelta', loss='binary_crossentropy', validation_set=None, n_init=1, train_on_init=False, default_activation_fn='relu'):
		self.training_set = training_set
		self.validation_set = validation_set
		h_index = len(layers) // 2
		encoded_dim = layers[h_index]

		# initialize n_init times and measure cost
		parameters_by_cost = {}
		for i in range(n_init):
			# Set up the layer structure
			X = Input(shape=(layers[0],))
			self.A = [X]
			for layer, units in enumerate(layers):
				if layer is 0: continue
				activation_fn = 'sigmoid' if layer == len(layers)-1 else default_activation_fn
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
			if train_on_init:
				self.model.fit(
					nb_epoch=100,
					batch_size=256,
					shuffle=True,
					verbose=0,
					x=self.training_set,
					y=self.training_set
				)
			parameters_by_cost[self.get_cost()] = self.model.get_weights()

		self.model.set_weights(parameters_by_cost[min(parameters_by_cost.keys())])
		self.model.compile(optimizer=optimizer, loss=loss)
		if n_init > 1:
			print('Choosing AE weights with cost', self.get_cost())
		del parameters_by_cost
		
		self.loss = loss

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

	def get_cost(self, data=None):
		if data is None:
			data = self.training_set
		return self.model.evaluate(data,data, verbose=0)

	def from_weights(weights, data):
		shapes = [w.shape for w in weights]
		layer_structure = [shape[0] for shape in shapes[::2]] + [shapes[-1][0]]
		ae = Autoencoder(layer_structure,data)
		ae.set_weights(weights)
		return ae