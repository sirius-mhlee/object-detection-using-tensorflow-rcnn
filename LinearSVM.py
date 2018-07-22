import numpy as np
import tensorflow as tf

import Configuration as cfg

class LinearSVM:
    def __init__(self, model, trainable):
        self.model = model
        self.var_dict = {}
        self.trainable = trainable

    def build(self, feature_holder, label_holder):
        self.weights, self.svm1 = self.svm_layer(feature_holder, 4096, cfg.object_class_num + 1, 'svm1')

        self.hinge_loss = tf.nn.relu(self.svm1 - label_holder + 1 - tf.reduce_sum(self.svm1 * label_holder, 1, keep_dims=True))
        self.hinge_loss_sum = tf.reduce_sum(self.hinge_loss, 1)
        self.hinge_loss_mean = tf.reduce_mean(self.hinge_loss)
        self.regularization = 0.5 * tf.reduce_sum(tf.square(self.weights))
        self.loss = self.regularization + 1.0 * self.hinge_loss_mean
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.svm1, 1), tf.argmax(label_holder, 1))
        self.accuracy_mean = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.model = None

    def get_var(self, initial_value, name, idx, var_name):
        if self.model is not None and name in self.model:
            value = self.model[name][idx]
        else:
            value = initial_value

        var = tf.Variable(value, name=var_name)

        self.var_dict[(name, idx)] = var

        return var

    def get_svm_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + '_weights')

        initial_value = tf.truncated_normal([out_size], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + '_biases')

        return weights, biases

    def svm_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_svm_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            svm = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return weights, svm

    def get_var_count(self):
        count = 0
        for var in list(self.var_dict.values()):
            count += np.multiply(var.get_shape().as_list())
        return count
