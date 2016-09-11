import tensorflow as tf
import numpy as np


class SOM:
	"""
	Efficient implementation of Kohonen Self-Organizing maps using Tensorflow.

	map_size_n: size of the (square) map
	num_expected_iterations: number of iterations to be used during training. This parameter is used to derive good parameters for the learning rate and the neighborhood radius.
	"""
	def __init__(self, input_shape, map_size_n, num_expected_iterations, session):
		input_shape = tuple([i for i in input_shape if i is not None])

		self.input_shape = input_shape
		self.sigma_act = tf.constant( 2.0*(reduce(lambda x, y:x*y, self.input_shape, 1)*0.05)**2, dtype=tf.float32 )

		self.n = map_size_n

		self.session = session

		self.alpha = tf.constant( 0.5 )
		self.timeconst_alpha = tf.constant( 2.0*num_expected_iterations/6.0) #2/6

		self.sigma = tf.constant( self.n/2.0 ) #self.n/2.0
		self.timeconst_sigma = tf.constant( 2.0*num_expected_iterations/5.0 ) #2/5


		self.num_iterations = 0
		self.num_expected_iterations = num_expected_iterations


		# Pre-initialize neighborhood function's data for efficiency
		self.row_indices = np.zeros((self.n, self.n))
		self.col_indices = np.zeros((self.n, self.n))
		for r in range(self.n):
			for c in range(self.n):
				self.row_indices[r, c] = r
				self.col_indices[r, c] = c

		self.row_indices = np.reshape(self.row_indices, [-1])
		self.col_indices = np.reshape(self.col_indices, [-1])

		## Compute d^2/2 for each pair of units, so that the neighborhood function can be computed as exp(-dist/sigma^2)
		self.dist = np.zeros((self.n*self.n, self.n*self.n))
		for i in range(self.n*self.n):
			for j in range(self.n*self.n):
				self.dist[i, j] = ( (self.row_indices[i]-self.row_indices[j])**2 + (self.col_indices[i]-self.col_indices[j])**2 )


		self.initialize_graph()


	def initialize_graph(self):
		self.weights = tf.Variable( tf.random_uniform((self.n*self.n, )+self.input_shape, 0.0, 1.0) )  ##TODO: match with input type, and check that my DQN implementation actually uses values in 0-255 vs 0-1?


		# 1) The first part computes the winning unit, potentially in batch mode. It only requires 'input_placeholder_' and 'current_iteration' to be filled in. This is used in get_batch_winner and get_batch_winner_activity, to be used in clustering novel vectors after training is complete.
		# The batch placeholder is not used in training, where only a single vector is supported at the moment.

		self.input_placeholder = tf.placeholder(tf.float32, (None,)+self.input_shape)
		self.current_iteration = tf.placeholder(tf.float32)

		## Compute the current iteration's neighborhood sigma and learning rate alpha:
		self.sigma_tmp = self.sigma * tf.exp( - self.current_iteration/self.timeconst_sigma  )
		self.sigma2 = 2.0*tf.mul(self.sigma_tmp, self.sigma_tmp)

		self.alpha_tmp = self.alpha * tf.exp( - self.current_iteration/self.timeconst_alpha  )


		self.input_placeholder_ = tf.expand_dims(self.input_placeholder, 1)
		self.input_placeholder_ = tf.tile(self.input_placeholder_, (1,self.n*self.n,1) )

		self.diff = self.input_placeholder_ - self.weights
		self.diff_sq = tf.square(self.diff)
		self.diff_sum = tf.reduce_sum( self.diff_sq, reduction_indices=range(2, 2+len(self.input_shape)) )

		# Get the index of the best matching unit
		self.bmu_index = tf.argmin(self.diff_sum, 1)

		self.bmu_dist = tf.reduce_min(self.diff_sum, 1)
		self.bmu_activity = tf.exp( -self.bmu_dist/self.sigma_act )


		self.diff = tf.squeeze(self.diff)



		## 2) The second part computes and applies the weight update. It requires 'diff_2' and 'dist_sliced' to be filled in. dist_sliced = self.dist[bmu_index, :]
		self.diff_2 = tf.placeholder(tf.float32, (self.n*self.n,)+self.input_shape)
		self.dist_sliced = tf.placeholder(tf.float32, (self.n*self.n,))

		self.distances = tf.exp(-self.dist_sliced / self.sigma2 )
		self.lr_times_neigh = tf.mul( self.alpha_tmp, self.distances )
		for i in range(len(self.input_shape)):
			self.lr_times_neigh = tf.expand_dims(self.lr_times_neigh, -1)
		self.lr_times_neigh = tf.tile(self.lr_times_neigh, (1,)+self.input_shape )

		self.delta_w = self.lr_times_neigh * self.diff_2

		self.update_weights = tf.assign_add(self.weights, self.delta_w)


	def train(self, input_x): #TODO: try training a batch all together, to optimize gpu usage?
		# Compute the winning unit
		bmu_index, diff = self.session.run([self.bmu_index, self.diff], {self.input_placeholder:[input_x], self.current_iteration:self.num_iterations})

		# Update the network's weights
		self.session.run(self.update_weights, {self.diff_2:diff, self.dist_sliced:self.dist[bmu_index[0],:], self.current_iteration:self.num_iterations})

		self.num_iterations = min(self.num_iterations+1, self.num_expected_iterations)


	def get_batch_winner(self, batch_input):
		"""
		Returns the index of the units in the network that best match each batch_input vector.
		"""
		indices = self.session.run([self.bmu_index], {self.input_placeholder:batch_input, self.current_iteration:self.num_iterations})

		return indices


	def get_batch_winner_activity(self, batch_input):
		"""
		Returns the activation value of the units in the network that best match each batch_input vector.
		"""
		activity = self.session.run([self.bmu_activity], {self.input_placeholder:batch_input, self.current_iteration:self.num_iterations})

		return activity


	def get_weights(self):
		"""
		Returns the full list of weights as [N*N, input_shape]
		"""
		return self.weights.eval()





## EXAMPLE : Color clustering

with tf.device("gpu:0"):
	sess = tf.InteractiveSession()

	num_training = 400
	s = SOM( (3,), 30, num_training, sess )

	sess.run(tf.initialize_all_variables())


	#For plotting the images
	from matplotlib import pyplot as plt

	#Training inputs for RGBcolors
	colors = np.array(
		 [[0., 0., 0.],
		  [0., 0., 1.],
		  [0., 0., 0.5],
		  [0.125, 0.529, 1.0],
		  [0.33, 0.4, 0.67],
		  [0.6, 0.5, 1.0],
		  [0., 1., 0.],
		  [1., 0., 0.],
		  [0., 1., 1.],
		  [1., 0., 1.],
		  [1., 1., 0.],
		  [1., 1., 1.],
		  [.33, .33, .33],
		  [.5, .5, .5],
		  [.66, .66, .66]])


	for i in range(num_training):
		rnd_ind = np.random.randint(0, len(colors))
		s.train(colors[rnd_ind,:])


	plt.imshow( np.reshape(s.get_weights(), [30, 30, 3]) )

	plt.show()


