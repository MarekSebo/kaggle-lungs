import tensorflow as tf
import numpy as np
import logging
import os
import cv2
logging.basicConfig(level=20)

class BuildModel(object):
    def __init__(self, name):
        self.weights = {}
        self.bias = {}
        self.num_w = 0
        self.num_convs = 0
        self.lstms = 0
        self.model_name = name
        self.learning_rate = 0.0001
        self.dir_ckpt = 'model/{}'.format(self.model_name)
        self.ones_in_batch_labels_perc = 0.1
        # self.graph = tf.Graph().as_default()

        self.build_new_model = False
        if tf.train.latest_checkpoint(self.dir_ckpt) is None:
            self.build_new_model = True
        else:
            self.restore_model()

    def data_shape(self, type, shape, sequences=False):
        # run this before the first layer
        # inputs:
        #   shape = shape bez batch size dimenzie
        #   type = typ premennej , napr tf.float32
        #   sequences = will seq lengths be used (e.g in texts with variable length)
        self.input = tf.placeholder(type, shape=(None, ) + shape)
        self.global_step = tf.Variable(0, name='global_step')
        self.out = self.input

        if sequences:
            self.seq_len = tf.placeholder(tf.int32, shape=[None])

    def step(self, x, y, sequence_lengths=None, lstm_state=None):
        # make one step
        feed = {self.input: x, self.labels: y}

        if sequence_lengths is not None:
            feed[self.seq_len] = sequence_lengths

        if lstm_state is not None:
            feed[self.c_state], feed[self.h_state] = lstm_state

        loss, _, predict, summary, step = self.session.run(
            [self.loss_op, self.optimizer_op, self.prediction, self.tb_loss_train, self.global_step], feed_dict=feed)

        # self.steps += 1
        self.file_writer.add_summary(summary, global_step=step)

        return loss, predict, step

    def predict(self, x, sequence_lengths=None, lstm_state=None):
        feed = {self.input: x}

        if sequence_lengths is not None:
            feed[self.seq_len] = sequence_lengths

        if lstm_state is not None:
            feed[self.c_state], feed[self.h_state] = lstm_state

        return self.session.run(self.prediction, feed_dict=feed)

    def valid(self, x, y):
        feed = {self.input: x, self.labels: y}
        loss, prediction, summary, step = self.session.run([self.loss_op, self.prediction, self.tb_loss_valid, self.global_step], feed_dict=feed)

        self.file_writer.add_summary(summary, global_step=step)
        return loss, prediction

    def conv2d(self, filter, num_filters, strides, padding, deconv=False):
        # define conv layer (init weights + graph)
        # inputs:
        #   filter = kernel size
        #   padding = {'SAME', 'VALID'}
        w_name = 'conv_{}'.format(self.num_convs)
        current_shape = self.out.get_shape()

        with tf.variable_scope('weights'):
            self.weights[w_name] = tf.get_variable(
                                w_name,
                                shape=(filter[0], filter[1], current_shape[-1], num_filters),
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False,
                                                                                        seed=None, dtype=tf.float32)
                                )
        with tf.variable_scope('biases'):
            self.bias[w_name] = tf.get_variable(
                w_name,
                shape=num_filters,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
            )

        if deconv:
            try:
                self.conv_filter[w_name] = filter
                self.conv_strides[w_name] = strides
                self.conv_padding[w_name] = padding
                self.conv_outputs[w_name] = self.out
            except:
                self.conv_filter = {w_name: filter}
                self.conv_strides = {w_name: strides}
                self.conv_padding = {w_name: padding}
                self.conv_outputs = {w_name: self.out}
                self.num_deconvs = 0

        self.out = tf.nn.conv2d(self.out,
                                filter=self.weights[w_name],
                                strides=(1, strides[0], strides[1], 1),
                                padding=padding,
                                name=w_name
                                )

        self.out = self.out + self.bias[w_name]

        logging.info('{}: {}'.format(w_name, self.out.get_shape()))

        self.num_convs += 1

    def deconv2d(self, u_net=False):
        # define deconv layer (init weights + graph)
        # inputs:
        #   filter = kernel size
        #   padding = {'SAME', 'VALID'}
        #   u_net = if true, you want to concatenate deconv output with corresponding encoder conv input
        w_name = 'deconv_{}'.format(self.num_convs - 1 - self.num_deconvs)
        w_conv_name = w_name[2:]

        current_shape = self.out.get_shape()
        output_shape = tf.pack([tf.shape(self.out)[0],
                                tf.shape(self.conv_outputs[w_conv_name])[1],
                                tf.shape(self.conv_outputs[w_conv_name])[2],
                                tf.shape(self.conv_outputs[w_conv_name])[3]])
        num_in_channels = current_shape[-1]
        num_out_channels = self.conv_outputs[w_conv_name].get_shape()[-1]

        # initialize variables
        with tf.variable_scope('weights'):
            self.weights[w_name] = tf.get_variable(
                w_name,
                shape=(self.conv_filter[w_conv_name][0], self.conv_filter[w_conv_name][1],
                       num_out_channels, num_in_channels),
                initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False, seed=None, dtype=tf.float32)
            )

        with tf.variable_scope('biases'):
            self.bias[w_name] = tf.get_variable(
                w_name,
                shape=num_out_channels,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
            )


        self.out = tf.nn.conv2d_transpose(self.out,
                                          filter=self.weights[w_name],
                                          output_shape=output_shape,
                                          strides=(1, self.conv_strides[w_conv_name][0],
                                                   self.conv_strides[w_conv_name][1], 1),
                                          padding=self.conv_padding[w_conv_name])
        self.out = self.out + self.bias[w_name]

        if u_net:
            self.out = tf.concat(3, [self.out, self.conv_outputs[w_conv_name]])

        logging.info('{}: {}'.format(w_name, self.out.get_shape()))
        self.num_deconvs += 1

    def fc(self, layer_size):
        current_shape = self.out.get_shape()
        self.out = tf.reshape(self.out, [-1, np.prod(current_shape[1:])])
        current_shape = self.out.get_shape()

        b_name = 'fc_bias_{}'.format(self.num_w)
        w_name = 'fc_weights_{}'.format(self.num_w)

        with tf.variable_scope('weights'):
            self.weights[w_name] = tf.get_variable(
                w_name,
                shape=(current_shape[-1], layer_size),
                initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
            )

        with tf.variable_scope('biases'):
            self.bias[b_name] = tf.get_variable(
                b_name,
                shape=layer_size,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
            )

        self.out = tf.matmul(self.out, self.weights[w_name])
        self.out = tf.add(self.out, self.bias[b_name])

        self.num_w += 1

    def lstm(self, lstm_size, output='all'):
        # build one LSTM layer
        # input:
        #   output = {'all' , 'last'}
        with tf.variable_scope('lstm_{}'.format(self.lstms)):
            self.c_state = tf.placeholder(dtype=tf.float32, shape=(None, lstm_size))
            self.h_state = tf.placeholder(dtype=tf.float32, shape=(None, lstm_size))
            tf.add_to_collection('states', self.c_state)
            tf.add_to_collection('states', self.h_state)
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.c_state, self.h_state)

            try:
                self.lstm_cells.append(tf.nn.rnn_cell.BasicLSTMCell(lstm_size))
            except NameError:
                self.lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(lstm_size)]

            self.out, self.lstm_state = tf.nn.dynamic_rnn(
                cell=self.lstm_cells[-1], inputs=self.out, dtype=tf.float32, initial_state=initial_state)

            if output == 'last':
                self.out = tf.reverse_sequence(self.out, self.seq_len, 1, batch_dim=0, name='rev_seq')
                self.out = self.out[:, 0, :]

            tf.add_to_collection('lstm_state', self.lstm_state[0])
            tf.add_to_collection('lstm_state', self.lstm_state[1])

        self.current_shape = self.out.get_shape().as_list()
        self.lstms += 1

    def relu(self):
        self.out = tf.nn.relu(self.out)

    def loss(self):
        # self.labels has shape (batch_size, h, w,2)
        # self.out  has shape (batch_size, h, w, 2)

        new_labels = self.labels[:,:,:,0]
        kernel = tf.ones((5, 5, 1, 1))

        new_labels = tf.expand_dims(new_labels, axis = -1)
        new_labels = tf.nn.conv2d(new_labels, kernel, (1,1,1,1), 'SAME')
        dilated_labels = tf.clip_by_value(new_labels, 0, 1)
        tensor_size = tf.cast(tf.size(dilated_labels), tf.float32)
        ones_in_dilated_labels_perc = tf.reduce_sum(dilated_labels) / tensor_size

        # vahy na loss
        w_1 = 1 / ones_in_dilated_labels_perc
        w_0 = 1 / (1 - ones_in_dilated_labels_perc)

        xent = tf.nn.softmax_cross_entropy_with_logits(self.out, self.labels)

        weights_per_label = dilated_labels * (w_1 - w_0) + w_0
        weights_per_label = tf.squeeze(weights_per_label)
        xent = tf.mul(xent, weights_per_label )

        loss_weighted = tf.reduce_mean(xent)
        return loss_weighted

    def embeddings(self, vocabulary_size, embedding_size, trainable=True):
        # same inputs as in TF
        self.vocabulary_size = vocabulary_size
        self.embeddings = tf.get_variable('embedding',
                                          shape=[vocabulary_size, embedding_size],
                                          initializer=tf.random_uniform_initializer(minval=0, maxval=None,
                                                                                    seed=None, dtype=tf.float32),
                                          trainable=trainable)
        self.out = tf.nn.embedding_lookup(self.embeddings, tf.cast(self.out, tf.int32))

    def optimizer(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)\
            .minimize(self.loss_op, global_step=self.global_step)

    def save(self):
        self.saver.save(self.session, 'model/{}/{}'.format(self.model_name, self.model_name),
                        global_step=self.global_step)
        self.saver.export_meta_graph('model/{}/graph.meta'.format(self.model_name))

    def load(self, dir_ckpt):
        self.saver = tf.train.import_meta_graph('model/{}/graph.meta'.format(self.model_name))
        self.saver.restore(self.session, tf.train.latest_checkpoint(dir_ckpt))

    def finish(self, learning_rate):
        # defines loss etc....
        self.prediction = tf.nn.softmax(self.out)
        self.labels = tf.placeholder(self.out.dtype, self.out.get_shape())

        self.loss_op = self.loss()
        self.optimizer_op = self.optimizer(learning_rate)

        self.tb_loss_train = tf.summary.scalar('loss_function_batch', self.loss_op)
        self.tb_loss_valid = tf.summary.scalar('loss_function_valid', self.loss_op)

        self.file_writer = tf.summary.FileWriter('model/{}'.format(self.model_name))

        # Warning!
        # do not change the "intput" to "input" if you want to use previously trained models, because the trained model
        # already contains part called intput and cannot be restored with "input"
        tf.add_to_collection('{}_output'.format(self.model_name), self.out)
        tf.add_to_collection('{}_input'.format(self.model_name), self.input)
        tf.add_to_collection('{}_labels'.format(self.model_name), self.labels)
        tf.add_to_collection('{}_prediction'.format(self.model_name), self.prediction)
        tf.add_to_collection('{}_others'.format(self.model_name), self.loss_op)
        tf.add_to_collection('{}_others'.format(self.model_name), self.optimizer_op)  # make it so that you can insert LR
        tf.add_to_collection('{}_others'.format(self.model_name), self.tb_loss_train)
        tf.add_to_collection('{}_others'.format(self.model_name), self.tb_loss_valid)
        tf.add_to_collection('{}_others'.format(self.model_name), self.global_step)

        self.session = tf.Session()
        self.initialize_model()

    def restore_model(self):
        # Warning!
        # do not change the intput to input if you want to use previously trained models, because the trained model
        # already contains part called intput and cannot be restored with "input"
        self.session = tf.Session()
        self.load(self.dir_ckpt)
        self.out = tf.get_collection('{}_output'.format(self.model_name))[0]
        self.input = tf.get_collection('{}_input'.format(self.model_name))[0]
        self.labels = tf.get_collection('{}_labels'.format(self.model_name))[0]
        self.prediction = tf.get_collection('{}_prediction'.format(self.model_name))[0]

        self.loss_op, self.optimizer_op, self.tb_loss_train, \
            self.tb_loss_valid, self.global_step = tf.get_collection('{}_others'.format(self.model_name))
        self.file_writer = tf.summary.FileWriter('model/{}'.format(self.model_name))
        logging.info('Model restored')

    def initialize_model(self):
        logging.info('File not found, initializing new model')
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    def save_and_close(self):
        self.save()
        self.session.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_and_close()


def accuracy(predictions, labels):
    # not working as TensorBoard log yet

    conf_matrix = np.zeros((2, 2))
    predictions = np.round(predictions)

    for i in range(2):
        for j in range(2):
            conf_matrix[1-i, 1-j] = np.sum(np.logical_and(predictions == j, labels == i))

    hitrate = 0
    if sum(conf_matrix[0, :]):
        hitrate = conf_matrix[0, 0] / sum(conf_matrix[0, :]) * 100

    precision = 0
    if sum(conf_matrix[:, 0]):
        precision = conf_matrix[0, 0] / sum(conf_matrix[:, 0]) * 100

    f1_score = 0
    if precision and hitrate:
        f1_score = 2 / (1/precision + 1/hitrate)

    logging.info('Hit-rate (found real positives): {}'.format(round(hitrate, ndigits=2)))
    logging.info('Precision (accuracy predicted positives): {}'.format(round(precision, ndigits=2)))
    logging.info('F1 score: {}'.format(round(f1_score, ndigits=2)))
    logging.info('Confusion matrix (real x predicted):\n{}'.format(conf_matrix))
    return conf_matrix
