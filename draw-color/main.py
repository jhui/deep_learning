import tensorflow as tf
import numpy as np
from dr_ops import *
from dr_utils import *
import input_data


class Draw():
    def __init__(self):
        # Read 55K of MNist training data + validation data + testing data
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.img_size = 28   # MNist is a 28x28 image
        self.N = 64          # Batch size used in the gradient descent.

        # LSTM configuration
        self.n_hidden = 256  # Dimension of the hidden state in each LSTM cell. (num_units in a TensorFlow LSTM cell)
        self.n_z = 10        # Dimension of the Latent vector
        self.T = 10          # Number of un-rolling time sequence in LSTM.

        # Attention configuration
        self.attention_n = 5 # Form a 5x5 grid for the attention.

        self.share_parameters = False  # Use in TensorFlow. Later we set to True so LSTM cell shares parameters.

        # Placeholder for images
        self.images = tf.placeholder(tf.float32, [None, 784])                  # image: 28 * 28 = 784

        # Create a random gaussian distrubtion we used to sample the latent variables (z).
        self.distrib = tf.random_normal((self.N, self.n_z), mean=0, stddev=1)  # (N, 10)

        # LSTM encoder and decoder
        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True)  # encoder Op
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True)  # decoder Op
        self.ct = [0] * self.T        # Image output at each time step (T, ...) -> (T, N, 784)

        # Mean, log siggma and signma used for each unroll time step.
        self.mu, self.logsigma, self.sigma = [0] * self.T, [0] * self.T, [0] * self.T

        # Initial state (zero-state) for LSTM.
        h_dec_prev = tf.zeros((self.N, self.n_hidden))  # Prev decoder hidden state (N, 256)
        enc_state = self.lstm_enc.zero_state(self.N, tf.float32) # (N, 256)
        dec_state = self.lstm_dec.zero_state(self.N, tf.float32) # (N, 256)

        x = self.images
        for t in range(self.T):

            # Calculate the input of LSTM cell with attention.
            # This is a function of
            #    the original image,
            #    the residual difference between previous output at the last time step and the original, and
            #    the hidden decoder state for the last time step.
            c_prev = tf.zeros((self.N, 784)) if t == 0 else self.ct[t - 1]  # (N, 784)
            x_hat = x - tf.sigmoid(c_prev)  # residual: (N, 784)
            r = self.read_attention(x, x_hat, h_dec_prev)        # (N, 50): (N, 25) for x and (N, 25) for x_hat

            # Using LSTM cell to encode the input with the encoder state
            # We use the attention input r and the previous decoder state as the input to the LSTM cell.
            self.mu[t], self.logsigma[t], self.sigma[t], enc_state = self.encode(enc_state, tf.concat([r, h_dec_prev], 1)) # (N, 10)

            # Sample from the distribution returned from the encoder to get z.
            z = self.sample(self.mu[t], self.sigma[t], self.distrib) # (N, 10)

            # Get the hidden decoder state and the cell state using the a LSTM decoder.
            h_dec, dec_state = self.decode_layer(dec_state, z) # (N, 256), (N, 256)

            # Calculate the output image at step t using attention with the decoder state as input.
            self.ct[t] = c_prev + self.write_attention(h_dec)

            # Update previous hidden state
            h_dec_prev = h_dec
            self.share_parameters = True  # from now on, share variables

        # Output the final output in the final timestep as the generated images
        self.generated_images = tf.nn.sigmoid(self.ct[-1])

        # Generation loss measure the difference between the generated images and the original.
        self.generation_loss = tf.reduce_mean(-tf.reduce_sum(
            self.images * tf.log(1e-10 + self.generated_images) + (1 - self.images) * tf.log(
                1e-10 + 1 - self.generated_images), 1))

        # Similar to the variation autoencoder, we add the KL divergence of the encoder distribution to the cost.
        kl_terms = [0] * self.T                # list of 10 elements: each element (N,)
        for t in range(self.T):
            mu2 = tf.square(self.mu[t])        # (N, 10)
            sigma2 = tf.square(self.sigma[t])  # (N, 10)
            logsigma = self.logsigma[t]        # (N, 10)
            kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - self.T * 0.5
        self.latent_loss = tf.reduce_mean(tf.add_n(kl_terms)) # Find mean of (N,)

        # All the generation loss and the latent loss and train the optimizer
        self.cost = self.generation_loss + self.latent_loss
        optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
        grads = optimizer.compute_gradients(self.cost)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)
        self.train_op = optimizer.apply_gradients(grads)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        for i in range(15000):
            xtrain, _ = self.mnist.train.next_batch(self.N)
            ct, gen_loss, lat_loss, _ = self.sess.run([self.ct, self.generation_loss, self.latent_loss, self.train_op],
                                                      feed_dict={self.images: xtrain})
            print("iter %d genloss %f latloss %f" % (i, gen_loss, lat_loss))
            if i % 500 == 0:
                ct = 1.0 / (1.0 + np.exp(-np.array(ct)))  # x_recons=sigmoid(canvas)
                for ct_iter in range(10):
                    results = ct[ct_iter]
                    results_square = np.reshape(results, [-1, 28, 28])
                    print(results_square.shape)
                    ims("results/" + str(i) + "-step-" + str(ct_iter) + ".jpg", merge(results_square, [8, 8]))

    def read_attention(self, x, x_hat, h_dec_prev):
        Fx, Fy, gamma = self.attn_window("read", h_dec_prev)     # (N, 5, 28),(N, 5, 28),(N,1)

        # we have the parameters for a patch of gaussian filters. apply them.
        def filter_img(img, Fx, Fy, gamma):
            Fxt = tf.transpose(Fx, perm=[0, 2, 1])               # (N, 28, 5)
            img = tf.reshape(img, [-1, self.img_size, self.img_size]) # (N, 28, 28)
            glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))                # (N, 5, 5)
            glimpse = tf.reshape(glimpse, [-1, self.attention_n ** 2])  # (N, 25)
            # finally scale this glimpse w/ the gamma parameter
            return glimpse * tf.reshape(gamma, [-1, 1])

        x = filter_img(x, Fx, Fy, gamma)                     # (N, 25)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)             # (N, 25)
        return tf.concat([x, x_hat], 1)

    # Given a hidden decoder layer: locate where to put attention filters
    def attn_window(self, scope, h_dec):
        # Use a linear network to compute the center point, sigma, distance for the grids.
        with tf.variable_scope(scope, reuse=self.share_parameters):
            parameters = dense(h_dec, self.n_hidden, 5)    # (N, 5)

        # gx_, gy_: center of 2d gaussian on a scale of -1 to 1
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(parameters, 5, 1)     # (N, 1)

        # move gx/gy to be a scale of -imgsize to +imgsize
        gx = (self.img_size + 1) / 2 * (gx_ + 1)   # (N, 1)
        gy = (self.img_size + 1) / 2 * (gy_ + 1)   # (N, 1)

        sigma2 = tf.exp(log_sigma2)  # (N, 1)

        # stride/delta: how far apart these patches will be
        delta = (self.img_size - 1) / ((self.attention_n - 1) * tf.exp(log_delta))   # (N, 1)

        # returns [Fx, Fy] Fx, Fy: (N, 5, 28)
        return self.filterbank(gx, gy, sigma2, delta) + (tf.exp(log_gamma),)

    # Given a center (gx, gy), sigma (sigma2) & distance between grid (delta)
    # Construct gaussian filter grids (5x5) represented by Fx = horiz. gaussian (N, 5, 28), Fy = vert. guassian (N, 5, 28)
    def filterbank(self, gx, gy, sigma2, delta):
        # Create 5 grid points around the center based on distance:
        grid_i = tf.reshape(tf.cast(tf.range(self.attention_n), tf.float32), [1, -1])  # (1, 5)
        mu_x = gx + (grid_i - self.attention_n / 2 - 0.5) * delta    # 5 grid points in x direction (N, 5)
        mu_y = gy + (grid_i - self.attention_n / 2 - 0.5) * delta

        mu_x = tf.reshape(mu_x, [-1, self.attention_n, 1])           # (N, 5, 1)
        mu_y = tf.reshape(mu_y, [-1, self.attention_n, 1])

        im = tf.reshape(tf.cast(tf.range(self.img_size), tf.float32), [1, 1, -1]) # (1, 1, 28)

        # list of gaussian curves for x and y
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])               # (N, 1, 1)
        Fx = tf.exp(-tf.square((im - mu_x) / (2 * sigma2)))   # (N, 5, 28) Filter weight for each grid point and x_i
        Fy = tf.exp(-tf.square((im - mu_y) / (2 * sigma2)))

        # normalize so area-under-curve = 1
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), 1e-8)    # (N, 5, 28)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), 1e-8)    # (N, 5, 28)
        return Fx, Fy

    # encode an attention patch
    def encode(self, prev_state, image):
        # update the RNN with image
        with tf.variable_scope("encoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_enc(image, prev_state)

        # map the RNN hidden state to latent variables
        with tf.variable_scope("mu", reuse=self.share_parameters):
            mu = dense(hidden_layer, self.n_hidden, self.n_z)
        with tf.variable_scope("sigma", reuse=self.share_parameters):
            logsigma = dense(hidden_layer, self.n_hidden, self.n_z)
            sigma = tf.exp(logsigma)
        return mu, logsigma, sigma, next_state

    def sample(self, mu, sigma, distrib):
        return mu + sigma * distrib

    def decode_layer(self, prev_state, latent):
        # update decoder RNN with latent var
        with tf.variable_scope("decoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_dec(latent, prev_state)

        return hidden_layer, next_state

    def write_attention(self, hidden_layer):
        with tf.variable_scope("writeW", reuse=self.share_parameters):
            w = dense(hidden_layer, self.n_hidden, self.attention_n ** 2)
        w = tf.reshape(w, [self.N, self.attention_n, self.attention_n])
        Fx, Fy, gamma = self.attn_window("write", hidden_layer)
        Fyt = tf.transpose(Fy, perm=[0, 2, 1])
        # [vert, attn_n] * [attn_n, attn_n] * [attn_n, horiz]
        wr = tf.matmul(Fyt, tf.matmul(w, Fx))
        wr = tf.reshape(wr, [self.N, self.img_size ** 2])
        return wr * tf.reshape(1.0 / gamma, [-1, 1])


model = Draw()
model.train()
