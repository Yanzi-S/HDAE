import tensorflow as tf


class Autoencoder(object):

    def __init__(self, n_layers, prior, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), ae_para=[0, 0]):
        self.n_layers = n_layers
        self.transfer = transfer_function
        self.in_keep_prob = 1 - ae_para[0]

        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.sparsity_level = 0.2  # np.repeat([0.05], self.n_hidden).astype(np.float32)
        self.sparse_reg = ae_para[1]
        self.balance_wei = ae_para[2]
        self.epsilon = 1e-06
        self.intrest_d = prior
        # model

        self.x = tf.placeholder(tf.float32, [None, self.n_layers[0]])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0,trainable=False)        

        self.hidden_encode = []
#        h = tf.cond(self.is_train, lambda: tf.nn.dropout(self.x, self.keep_prob), lambda: self.x)
#        h = tf.cond(self.is_train, lambda: tf.add(self.x, self.keep_prob * tf.random_normal(shape=(self.n_layers[0], ))), lambda: self.x)
        h = tf.nn.dropout(self.x, self.keep_prob) * self.keep_prob
#        h = tf.add(self.x, self.keep_prob * tf.random_normal(shape=(self.n_layers[0], )))
        self.noisy_x = h
        for layer in range(len(self.n_layers)-1):
#            h = self.transfer(
#                tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
#                       self.weights['encode'][layer]['b']))
            h = self.transfer(
                tf.matmul(h, self.weights['encode'][layer]['w']))
            self.hidden_encode.append(h)

        self.hidden_recon = []
        for layer in range(len(self.n_layers)-1):
#            h = self.transfer(
#                tf.add(tf.matmul(h, self.weights['recon'][layer]['w']),
#                       self.weights['recon'][layer]['b']))
            h = self.transfer(
                tf.matmul(h, self.weights['recon'][layer]['w']))
            self.hidden_recon.append(h)
        self.reconstruction = self.hidden_recon[-1]

        self.sigma1 = tf.exp(-tf.reduce_sum(tf.pow(tf.subtract(self.x, self.intrest_d), 2.0), axis=-1))
#        self.sigma2 = 1-tf.exp(-tf.reduce_sum(tf.multiply(self.x, self.intrest_d),axis=-1)/(tf.multiply(tf.norm(self.x,axis=-1), tf.norm(self.intrest_d,axis=-1))+0.00001))
        self.sigma2 = tf.pow(tf.reduce_sum(tf.multiply(self.x, self.intrest_d),axis=-1)/(tf.multiply(tf.norm(self.x,axis=-1), tf.norm(self.intrest_d,axis=-1))+0.00001),1)
#180/3.1415926*
        self.cost_mse = tf.reduce_mean(self.sigma1*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0), axis=-1))
        self.cost_sam = tf.reduce_mean(self.sigma2*tf.acos(tf.reduce_sum(tf.multiply(self.reconstruction, self.x),axis=-1)/(tf.multiply(tf.norm(self.reconstruction,axis=-1), tf.norm(self.x,axis=-1))+0.00001)))
#        self.cost_lr = tf.trace(tf.sqrt(tf.matmul(tf.transpose(self.hidden_encode[-1]),self.hidden_encode[-1])))
#        self.cost_lr = tf.reduce_sum(tf.reduce_sum(self.hidden_encode[-1],-1)-1,2)
#        self.cost_lr = tf.reduce_sum(tf.pow(tf.reduce_sum(self.hidden_encode[-1],-1) - 1, 2))
        self.cost_lr = self.kl_divergence(self.sparsity_level, self.hidden_encode[-1])

        # cost
        if self.sparse_reg == 0:
            self.cost = self.balance_wei * self.cost_mse + self.cost_sam
        else:
            self.cost = self.balance_wei * self.cost_mse + self.cost_sam + self.sparse_reg * self.cost_lr
            tf.summary.scalar('cost_lr', self.cost_lr)
#            self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))+\
#                        self.sparse_reg * self.kl_divergence(self.sparsity_level, self.hidden_encode[-1])
#                        self.sparse_reg * self.kl_divergence(self.sparsity_level, self.hidden_encode[-1])
#                        self.sparse_reg * tf.norm(tf.matmul(self.weights['recon'][0]['w'],tf.transpose(self.weights['recon'][0]['w']))-tf.multiply(tf.matmul(self.weights['recon'][0]['w'],tf.transpose(self.weights['recon'][0]['w'])),tf.eye(self.n_layers[1])))+\
#            self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))+\
#                        0.5 * tf.reduce_sum(tf.acos(tf.reduce_sum(tf.multiply(self.reconstruction, self.x),axis=-1)/(tf.norm(self.reconstruction,axis=-1) * tf.norm(self.x,axis=-1))))+\
#                        0.5 * tf.norm(tf.matmul(self.weights['recon'][0]['w'],tf.transpose(self.weights['recon'][0]['w']))-tf.multiply(tf.matmul(self.weights['recon'][0]['w'],tf.transpose(self.weights['recon'][0]['w'])),tf.eye(self.n_layers[1])))+\
#                        self.sparse_reg * self.kl_divergence(self.sparsity_level, self.hidden_encode[-1])

        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('cost_sam', self.cost_sam)
        tf.summary.scalar('cost_mse', self.cost_mse)

        self.optimizer = optimizer.minimize(self.cost,global_step=self.global_step)
        self.merged = tf.summary.merge_all()
 
    def _initialize_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        # Encoding network weights
        encoder_weights = []
        for layer in range(len(self.n_layers)-1):
            w = tf.Variable(
                initializer((self.n_layers[layer], self.n_layers[layer + 1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer + 1]], dtype=tf.float32))
            encoder_weights.append({'w': w, 'b': b})
        # Recon network weights
        recon_weights = []
        for layer in range(len(self.n_layers)-1, 0, -1):
            w = tf.Variable(
                initializer((self.n_layers[layer], self.n_layers[layer - 1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer - 1]], dtype=tf.float32))
            recon_weights.append({'w': w, 'b': b})
        all_weights['encode'] = encoder_weights
        all_weights['recon'] = recon_weights
        return all_weights


    def kl_divergence(self, p, p_hat):
        return tf.reduce_mean(p * tf.log(tf.clip_by_value(p, 1e-8, tf.reduce_max(p)))
                              - p * tf.log(tf.clip_by_value(p_hat, 1e-8, tf.reduce_max(p_hat)))
                              + (1 - p) * tf.log(tf.clip_by_value(1-p, 1e-8, tf.reduce_max(1-p)))
                              - (1 - p) * tf.log(tf.clip_by_value(1-p_hat, 1e-8, tf.reduce_max(1-p_hat))))

    def partial_fit(self):
        return  (self.cost, self.optimizer)

    def calc_total_cost(self):
        return self.cost

    def show_noisy_x(self):
        return self.noisy_x
    def sigma(self):
        return self.sigma1,self.sigma2
    
    def transform(self):
        return self.hidden_encode[-1]

    def reconstruct(self):
        return self.reconstruction

    def setNewX(self,x):
        self.hidden_encode = []
        h = tf.nn.dropout(x, self.keep_prob)
        for layer in range(len(self.n_layers) - 1):
            h = self.transfer(
                tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
                       self.weights['encode'][layer]['b']))
            self.hidden_encode.append(h)

    def cost_summary(self):
        return self.merged
 

