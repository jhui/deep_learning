import os
import math
import random
import numpy as np
import tensorflow as tf

class MemN2N(object):
    def __init__(self, config, sess):
        self.nwords = config.nwords         # 10,000
        self.init_u = config.init_u         # 0.1 (We don't need a query in language model. So set u to be 0.1
        self.init_std = config.init_std     # 0.05
        self.batch_size = config.batch_size # 128
        self.nepoch = config.nepoch         # 100
        self.nhop = config.nhop             # 6
        self.edim = config.edim             # 150
        self.mem_size = config.mem_size     # 100
        self.lindim = config.lindim         # 75
        self.max_grad_norm = config.max_grad_norm   # 50

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        # (?, 150) Unlike Q&A, the language model do not need a query (or care what is its value).
        # So we bypass q and fill u directly with 0.1 later.
        self.u = tf.placeholder(tf.float32, [None, self.edim], name="u")

        # (?, 100) Sec. 4.1, we add temporal encoding to capture the time sequence of the memory Xi.
        self.T = tf.placeholder(tf.int32, [None, self.mem_size], name="T")

        # (N, 10000) The answer word we want. (Next word for the language model)
        self.target = tf.placeholder(tf.float32, [self.batch_size, self.nwords], name="target")

        # (N, 100) The memory Xi. For each sentence here, it contains 1 single word only.
        self.sentences = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="sentences")

        # Store the value of u at each layer
        self.u_s = []
        self.u_s.append(self.u)

        self.lr = None
        self.current_lr = config.init_lr       # learning rate
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
        self.global_step = tf.Variable(0, name="global_step")

        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std)) # Embedding A for sentences
        self.C = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std)) # Embedding C for sentences
        self.H = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))   # Multiple it with u before adding to o

        # Sec 4.1: Temporal Encoding to capture the time order of the sentences.
        self.T_A = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))
        self.T_C = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))

        # Sec 2: We are using layer-wise (RNN-like) which the embeddings for each layers are sharing the parameters.
        # (N, 100, 150) m_i = sum A_ij * x_ij + T_A_i
        m_a = tf.nn.embedding_lookup(self.A, self.sentences)
        m_t = tf.nn.embedding_lookup(self.T_A, self.T)
        m = tf.add(m_a, m_t)

        # (N, 100, 150) c_i = sum C_ij * x_ij + T_C_i
        c_a = tf.nn.embedding_lookup(self.C, self.sentences)
        c_t = tf.nn.embedding_lookup(self.T_C, self.T)
        c = tf.add(c_a, c_t)

        for h in range(self.nhop):
            u = tf.reshape(self.u_s[-1], [-1, 1, self.edim])
            scores = tf.matmul(u, m, adjoint_b=True)
            scores = tf.reshape(scores, [-1, self.mem_size])

            P = tf.nn.softmax(scores)     # (N, 100)
            P = tf.reshape(P, [-1, 1, self.mem_size])

            o = tf.matmul(P, c)
            o = tf.reshape(o, [-1, self.edim])

            # Section 2: We are using layer-wise (RNN-like), so we multiple u with H.
            uh = tf.matmul(self.u_s[-1], self.H)
            next_u = tf.add(uh, o)

            if self.lindim == self.edim:
                self.u_s.append(next_u)
            elif self.lindim == 0:
                self.u_s.append(tf.nn.relu(next_u))
            else:
                # Section 5:  To aid training, we apply ReLU operations to half of the units in each layer.
                F = tf.slice(next_u, [0, 0], [self.batch_size, self.lindim])
                G = tf.slice(next_u, [0, self.lindim], [self.batch_size, self.edim-self.lindim])
                K = tf.nn.relu(G)
                self.u_s.append(tf.concat(axis=1, values=[F, K]))

    def build_model(self):
        # Build MemN2N
        self.build_memory()

        # Build the network to output the final word.
        self.W = tf.Variable(tf.random_normal([self.edim, self.nwords], stddev=self.init_std))
        z = tf.matmul(self.u_s[-1], self.W)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=self.target)

        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.A, self.C, self.H, self.T_A, self.T_C, self.W]
        grads_and_vars = self.opt.compute_gradients(self.loss, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                   for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def train(self, data):
        n_batch = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        u = np.ndarray([self.batch_size, self.edim], dtype=np.float32)      # (N, 150) Will fill with 0.1
        T = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)    # (N, 100) Will fill with 0..99
        target = np.zeros([self.batch_size, self.nwords])                   # one-hot-encoded
        sentences = np.ndarray([self.batch_size, self.mem_size])

        u.fill(self.init_u)   # (N, 150) Fill with 0.1 since we do not need query in the language model.
        for t in range(self.mem_size):   # (N, 100) 100 memory cell with 0 to 99 time sequence.
            T[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=n_batch)

        for idx in range(n_batch):
            if self.show:
                bar.next()
            target.fill(0)      # (128, 10,000)
            for b in range(self.batch_size):
                # We random pick a word in our data and use that as the word we need to predict using the language model.
                m = random.randrange(self.mem_size, len(data))
                target[b][data[m]] = 1                       # Set the one hot vector for the target word to 1

                # (N, 100). Say we pick word 1000, we then fill the memory using words 1000-150 ... 999
                # We fill Xi (sentence) with 1 single word according to the word order in data.
                sentences[b] = data[m - self.mem_size:m]

            _, loss, self.step = self.sess.run([self.optim,
                                                self.loss,
                                                self.global_step],
                                                feed_dict={
                                                    self.u: u,
                                                    self.T: T,
                                                    self.target: target,
                                                    self.sentences: sentences})
            cost += np.sum(loss)

        if self.show:
            bar.finish()
        return cost/n_batch/self.batch_size

    def test(self, data, label='Test'):
        n_batch = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        u = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        T = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        sentences = np.ndarray([self.batch_size, self.mem_size])

        u.fill(self.init_u)
        for t in range(self.mem_size):
            T[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar(label, max=n_batch)

        m = self.mem_size
        for idx in range(n_batch):
            if self.show:
                bar.next()
            target.fill(0)
            for b in range(self.batch_size):
                target[b][data[m]] = 1
                sentences[b] = data[m - self.mem_size:m]
                m += 1

                if m >= len(data):
                    m = self.mem_size

            loss = self.sess.run([self.loss], feed_dict={self.u: u,
                                                         self.T: T,
                                                         self.target: target,
                                                         self.sentences: sentences})
            cost += np.sum(loss)

        if self.show:
            bar.finish()
        return cost/n_batch/self.batch_size

    def run(self, train_data, test_data):
        if not self.is_test:
            for idx in range(self.nepoch):
                train_loss = np.sum(self.train(train_data))
                test_loss = np.sum(self.test(test_data, label='Validation'))

                # Logging
                self.log_loss.append([train_loss, test_loss])
                self.log_perp.append([math.exp(train_loss), math.exp(test_loss)])

                state = {
                    'perplexity': math.exp(train_loss),
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'valid_perplexity': math.exp(test_loss)
                }
                print(state)

                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx-1][1] * 0.9999:
                    self.current_lr = self.current_lr / 1.5
                    self.lr.assign(self.current_lr).eval()
                if self.current_lr < 1e-5:
                    break

                if idx % 10 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step = self.step.astype(int))
        else:
            self.load()

            valid_loss = np.sum(self.test(train_data, label='Validation'))
            test_loss = np.sum(self.test(test_data, label='Test'))

            state = {
                'valid_perplexity': math.exp(valid_loss),
                'test_perplexity': math.exp(test_loss)
            }
            print(state)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")
