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

def get_model_fn(hparams):
    def model_fn(features, labels, mode):
        vocab_size_output = len([1 for l in open(hparams['vocabulary_path'])])
        answer_embeddings = tf.get_variable(
             'answer_embeddings',
                [vocab_size_output, hparams['embedding_size']])
        source_embeddings = answer_embeddings
        
        if 'so' in features:
            # copy input
            features_so = tf.cast(features['so'], tf.int32, name='so')
            features_so = tf.reshape(features_so, [-1, tf.shape(features_so)[-1]])
            batch_size = tf.shape(features_so)[0]
            source_len = length_int1(features_so)
            features_so = features_so[:, :tf.reduce_max(source_len)]
            tf.shape(features_so, name='stripped_shape')
            source = tf.nn.embedding_lookup(source_embeddings, features_so)
            dropout_bool = hparams['dropout_rate'] is not None and mode != tf.estimator.ModeKeys.PREDICT
            
            # embedding of labels
            labels_embedded = tf.concat([tf.zeros([batch_size, 1, hparams['embedding_size']],
                    dtype=tf.float32), source], 1) # padding first token
            if hparams['word_dropout'] and mode == tf.estimator.ModeKeys.TRAIN:
                wdrop_mask = tf.random_uniform([batch_size, tf.reduce_max(source_len) + 1, 1])
                wdrop_mask = (tf.sign(wdrop_mask - hparams['word_dropout']) + 1) / 2
                wdrop_mask = tf.cast(wdrop_mask, tf.float32)
                labels_embedded *= wdrop_mask
            
            # Encode
            layers = [tf.contrib.rnn.DropoutWrapper(
                            tf.contrib.rnn.GRUCell(
                                    hparams['hidden_size'], name='enc_cell%i' % i),
                                input_keep_prob=1 - hparams['dropout_rate'] if dropout_bool else 1
                            ) for i in range(hparams['num_layers'])]
            enc_cell = tf.contrib.rnn.MultiRNNCell(layers)
            context, encoder_state = tf.nn.dynamic_rnn(
                enc_cell,
                source,
                dtype=tf.float32,
                sequence_length=source_len
            )
        
        # variational latent layer
        if 'mu_sigma' in features:
            mu_sigma = tf.reshape(features['mu_sigma'], [-1, hparams['latent_size'] * 2])
            mu, sigma = tf.split(mu_sigma, 2, axis=1)
            batch_size = tf.shape(mu)[0]
            _ = tf.layers.dense(tf.zeros([batch_size, hparams['hidden_size'] * hparams['num_layers']]),
                            2 * hparams['latent_size'])
            dropout_bool = False
            logsigma = tf.log(sigma)
        else:
            x = tf.concat(encoder_state, 1)
            mu, logsigma = tf.split(tf.layers.dense(x, 2 * hparams['latent_size']), 2, axis=1)
            sigma = tf.exp(logsigma)
            mu_sigma = tf.concat([mu, sigma], 1)
        z = mu + sigma * tf.random_normal([batch_size, hparams['latent_size']])
        y = tf.layers.dense(z, hparams['num_layers'] * hparams['hidden_size'])
        to_decoder = tuple(tf.split(y, hparams['num_layers'], axis=1))
        
        # calculate kullback-leibler divergence
        loss_kl = (tf.reduce_mean(mu**2) + tf.reduce_mean(sigma**2) - 2 * tf.reduce_mean(logsigma) - 1) / 2
        loss_kl *= hparams['latent_size']
        loss_kl = tf.identity(loss_kl, name='loss_kl')
        
        # make initial state of decoder
        initial_state = to_decoder
        if mode == tf.estimator.ModeKeys.PREDICT and hparams['use_beam_search']:
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                    initial_state, multiplier=hparams['beam_width'])
        
        # Decode
        # make decoder cell
        layers = [tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.GRUCell(
                                hparams['hidden_size'],
                                name='dec_cell%i' % i
                        ),
                        input_keep_prob=1 - hparams['dropout_rate'] if dropout_bool else 1
                ) for i in range(hparams['num_layers'])]
        dec_cell = tf.contrib.rnn.MultiRNNCell(layers)
        
        # choose projection layer
        if mode == tf.estimator.ModeKeys.PREDICT and hparams['use_beam_search']:
            projection_layer = layers_core.Dense(vocab_size_output,
                            use_bias=False,
                            activation=lambda x: tf.log(tf.nn.softmax(x) + 1e-8))
        else:
            projection_layer = layers_core.Dense(vocab_size_output,
                            use_bias=False,
                            activation=tf.nn.softmax)
        # decoding at inference
        if mode == tf.estimator.ModeKeys.PREDICT:
            # Decoder
            if hparams['use_beam_search']:
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=dec_cell,
                        embedding=answer_embeddings,
                        start_tokens=tf.fill([batch_size], 0),
                        end_token=1,
                        initial_state=decoder_initial_state,
                        beam_width=hparams['beam_width'],
                        output_layer=projection_layer,
                        length_penalty_weight=hparams['length_penalty_weight'])
            else:
                # Helper
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        answer_embeddings,
                        tf.fill([batch_size], 0), 1)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                        dec_cell, helper, initial_state,
                        output_layer=projection_layer)
            # Dynamic decoding
            if 'so' in features:
                max_answer_len = tf.reduce_max(source_len) * 2
            else:
                max_answer_len = hparams['max_sentence_length']
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=max_answer_len)
            if hparams['use_beam_search']:
                classes = outputs.predicted_ids[:, :, 0]
                classes_topn = outputs.predicted_ids
                beam_scores = outputs.beam_search_decoder_output.scores
            else:
                classes = outputs.sample_id
        
        # decoding at training / evaluation
        if mode != tf.estimator.ModeKeys.PREDICT:
            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                    labels_embedded, source_len, time_major=False)
            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                    dec_cell, helper, initial_state,
                    output_layer=projection_layer)
            # Dynamic decoding
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            classes = outputs.sample_id
            probabilities = outputs.rnn_output
        
        # form predictions list
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {}
            predictions['classes'] = classes
            if hparams['use_beam_search']:
                predictions['classes_topn'] = classes_topn
                predictions['beam_scores'] = beam_scores
            print('predictions:', predictions)
            if 'so' in features:
                predictions['mu_sigma'] = mu_sigma
                return tf.estimator.EstimatorSpec(mode=mode,
                        predictions=predictions,
                        export_outputs={'output':tf.estimator.export.PredictOutput(mu_sigma)})
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                        predictions=predictions,
                        export_outputs={'output':tf.estimator.export.PredictOutput(classes)})
        
        # Calc answer cross-entropy
        labels = features_so + tf.concat([tf.ones([batch_size, 1], dtype=tf.int32),
                    tf.sign(features_so[:, :-1])], axis=1) * (1 - tf.sign(features_so)) # <EOS>
        answer_ref = tf.one_hot(labels, vocab_size_output)
        if 'label_smoothing' in hparams:
            eps = hparams['label_smoothing']
            answer_ref = (answer_ref + eps / (vocab_size_output - 1)) - answer_ref * (eps + eps / (vocab_size_output - 1))
        cross_entropy = tf.log(probabilities + 1e-8) * answer_ref
        cross_entropy = -tf.reduce_sum(cross_entropy, axis=2)
        answer_mask = tf.cast(tf.sign(labels), tf.float32)
        cross_entropy *= answer_mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        loss_ce = tf.reduce_mean(cross_entropy)
        
        # Calc summary loss
        loss = loss_ce + loss_kl
        loss = tf.identity(loss, name='loss')
        
        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            if hparams['adam']:
                optimizer = tf.train.AdamOptimizer(learning_rate=hparams['learning_rate'])
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=hparams['learning_rate'])
            # gradient clipping
            gvs = optimizer.compute_gradients(loss)
            if hparams['clip_grad_norm']:
                gvs = [(grad if grad is None else tf.clip_by_norm(grad, hparams['clip_grad_norm']),
                        var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(gvs, tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {}
        eval_metric_ops["accuracy_words"] = tf.metrics.accuracy(
                labels=features_so, predictions=classes)
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    return model_fn

def length_int1(sequence, to_int=True, name=None):
    used = tf.sign(sequence)
    length = tf.reduce_sum(used, 1, name=name)
    length = tf.cast(length, tf.int32)
    return length













