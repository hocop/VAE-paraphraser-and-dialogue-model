import copy
from tensorflow.python.layers import core as layers_core
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import *
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

from my_att_cell import AttCopyWrapper

def get_model_fn(hparams):
    def model_fn(features, labels, mode):
        # copy input
        mu_sigma_in = tf.reshape(features['mu_sigma'], [-1, hparams['latent_size'] * 2])
        mu_in, sigma_in = tf.split(mu_sigma_in, 2, axis=1)
        # get parameters from data
        batch_size = tf.shape(mu_in)[0]
        dropout_bool = hparams['dropout_rate'] is not None and mode != tf.estimator.ModeKeys.PREDICT
        
        # sample from distribution
        if mode == tf.estimator.ModeKeys.TRAIN:
            z_in = mu_in + sigma_in * tf.random_normal([batch_size, hparams['latent_size']])
        else:
            z_in = mu_in
        
        # layers of dialogue network
        x = tf.layers.dense(z, hparams['hidden_size'], use_bias=False, name='initial_projection')
        for i_layer in range(hparams['num_layers']):
            x = x + tf.layers.dense(x, hparams['hidden_size'], activation=tf.nn.relu, name='layer_%i' % i_layer)
            x = x / np.sqrt(2)
            if dropout_bool:
                x = tf.nn.dropout(x, keep_prob=1 - hparams['dropout_rate'])
        x = tf.layers.dense(x, 2 * hparams['latent_size'], name='final projection')
        mu, logsigma = tf.split(x, 2, axis=1)
        sigma = tf.exp(logsigma)
        mu_sigma = tf.concat([mu, sigma], axis=1)
        
        # form predictions list
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'mu_sigma': mu_sigma,
            }
        
        # Calc answer cross-entropy
        mu_sigma_label = tf.reshape(labels, [-1, hparams['latent_size'] * 2])
        mu_label, sigma_label = tf.split(mu_sigma_label, 2, axis=1)
        loss_ce = (
                    tf.reduce_mean(tf.square(sigma_label / sigma)) + 
                    tf.reduce_mean(tf.square((mu - mu_label) / sigma)) + 
                    2 * tf.reduce_mean(tf.log(sigma))
            )
        # Calc summary loss
        loss = loss_ce
        loss = tf.identity(loss, name='loss')
        
        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            if hparams['adam']:
                optimizer = tf.train.AdamOptimizer(learning_rate=hparams['learning_rate'])
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=hparams['learning_rate'])
            train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {}
        eval_metric_ops["accuracy_in_1_sigma"] =  tf.metrics.accuracy(
                        labels=tf.ones([batch_size], dtype=tf.int32),
                        predictions=(tf.sign(1 - tf.reduce_sum(tf.square((mu - mu_label) / sigma_label),
                                                                axis=1)) + 1) / 2)
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    return model_fn

    

















