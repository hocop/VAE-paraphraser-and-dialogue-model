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

from multiencoder_utils import Multiencoder

def get_model_fn(hparams):
    def model_fn(features, labels, mode):
        vocab_size_output = sum([1 for l in open(hparams['output_encoder']['path'])])
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
            dropout_bool = hparams['dropout_rate'] is not None and mode != tf.estimator.ModeKeys.PREDICT and not hparams['freezing_mode']
            
            # embed labels
            if mode != tf.estimator.ModeKeys.PREDICT or hparams['freezing_mode']:
                labels = tf.cast(tf.reshape(labels, [-1, tf.shape(labels)[-1]]), tf.int32)
                ans_len = length_int1(labels)
                max_ans_len = tf.reduce_max(ans_len)
                labs = labels[:,:tf.reduce_max(ans_len)]
                answer_mask = tf.sign(labs, name='answer_mask')
                answer_mask = tf.cast(answer_mask, tf.float32)
                features_help = tf.concat([tf.zeros([batch_size, 1], dtype=tf.int32), labels], 1)
                labels_embedded = tf.nn.embedding_lookup(answer_embeddings,
                        features_help[:,:tf.reduce_max(ans_len) + 1])
            
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
        loss_kl = (tf.reduce_mean(mu**2) + tf.reduce_mean(sigma**2) - 2 * tf.reduce_mean(logsigma) - 1) / 2
        loss_kl = tf.identity(loss_kl, name='loss_kl')
        
        # Decode
        # make decoder cell
        layers = [tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.GRUCell(
                                hparams['hidden_size'], name='dec_cell%i' % i),
                            input_keep_prob=1 - hparams['dropout_rate'] if dropout_bool else 1
                        ) for i in range(hparams['num_layers'])]
        dec_cell = tf.contrib.rnn.MultiRNNCell(layers)
        
        # make initial state
        initial_state = to_decoder
        if mode == tf.estimator.ModeKeys.PREDICT and hparams['use_beam_search']:
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                    initial_state, multiplier=hparams['beam_width'])
        
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
        if mode == tf.estimator.ModeKeys.PREDICT or hparams['freezing_mode']:
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
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=hparams['max_answer_len'])
            if hparams['use_beam_search']:
                classes = outputs.predicted_ids[:, :, 0]
                classes_topn = outputs.predicted_ids
                beam_scores = outputs.beam_search_decoder_output.scores
            else:
                classes = outputs.sample_id
        
        # decoding at training / evaluation
        if mode != tf.estimator.ModeKeys.PREDICT or hparams['freezing_mode']:
            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                    labels_embedded, ans_len, time_major=False)
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
            if hparams['copy']:
                predictions['p_gens'] = p_gens
            if hparams['attention_scheme'] != 'None':
                predictions['attention_image'] = att_image
            print('predictions:', predictions)
            if 'so' in features:
                return tf.estimator.EstimatorSpec(mode=mode,
                        predictions=predictions,
                        export_outputs={'output':tf.estimator.export.PredictOutput(mu_sigma)})
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                        predictions=predictions,
                        export_outputs={'output':tf.estimator.export.PredictOutput(classes)})
        
        # Calc answer cross-entropy
        answer_ref = tf.one_hot(labs, vocab_size_output)
        if 'label_smoothing' in hparams:
            eps = hparams['label_smoothing']
            answer_ref = (answer_ref + eps / (vocab_size_output - 1)) - answer_ref * (eps + eps / (vocab_size_output - 1))
        cross_entropy = tf.log(probabilities + 1e-8) * answer_ref
        cross_entropy = -tf.reduce_sum(cross_entropy, axis=2)
        answer_mask = tf.stop_gradient(answer_mask)
        cross_entropy *= answer_mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        if hparams['loss_normalize_by_length']:
            ans_norm = tf.reshape(tf.reduce_sum(answer_mask, 1), [-1, 1])
            cross_entropy /= tf.cast(ans_norm, tf.float32)
        loss_ce = tf.reduce_mean(cross_entropy)
        if 'decoder_speedup' in hparams and hparams['decoder_speedup'] > 1:
            loss_ce /= hparams['decoder_speedup']
        
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
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {}
        if 'thinker' in hparams and hparams['thinker']['enabled']:
            eval_metric_ops["accuracy_binary"] =  tf.metrics.accuracy(
                        labels=tf.cast(binary_ref, tf.int32),
                        predictions=(tf.sign(sigma_x - 0.5) + 1) / 2)
        eval_metric_ops["accuracy_words"] = tf.metrics.accuracy(
                labels=labs, predictions=classes)
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    return model_fn

def length_int1(sequence, to_int=True, name=None):
            used = tf.sign(sequence)
            length = tf.reduce_sum(used, 1, name=name)
            length = tf.cast(length, tf.int32)
            return length

class LambdaLayer(layers_core.base.Layer):
    def __init__(self, function=tf.identity, name=None):
        super(LambdaLayer, self).__init__(name=name)
        self.function = function
    def compute_output_shape(self, input_shape):
        return input_shape
    def build(self, inputs_shape):
        pass
    def call(self, inputs):
        return self.function(inputs, name=self.name)

class SumWithTriggersCell(rnn_cell_impl.RNNCell):
    def __init__(self,
                dim,
                reset,
                return_current=False,
                positional_encoding=False,
                name=None):
        super(SumWithTriggersCell, self).__init__(name=name)
        self.dim = dim
        self.reset = reset
        self.return_current = return_current
        self.positional_encoding = positional_encoding
    @property
    def state_size(self):
        return self.dim
    @property
    def output_size(self):
        return self.dim
    def build(self, inputs_shape):
        if self.positional_encoding:
            self.k_column = tf.range(1, self.dim + 1, dtype=tf.float32) / self.dim
            self.k_column = tf.reshape(self.k_column, [1, self.dim]) # column vector VEC_k = (k / dim)
    
    def call(self, inputs, state):
        trigger = tf.reshape(inputs[:, 0], [-1, 1]) # this is 0 or 1
        if self.positional_encoding:
            pe_index = tf.reshape(inputs[:, 1], [-1, 1]) # this is normed (0, 1)
            vector = inputs[:, 2:]
            l = (1 - pe_index) - self.k_column * (1 - 2 * pe_index)
            vector = vector * l
        else:
            vector = inputs[:, 1:]
        vector = tf.reshape(vector, [-1, self.dim])
        cur_sum = state + vector
        if self.reset:
            if self.return_current:
                next_state = state * (1 - tf.sign(trigger)) + vector
            else:
                next_state = cur_sum * (1 - tf.sign(trigger))
        else:
            next_state = cur_sum
        if self.return_current:
            out = state + vector * tf.sign(trigger)
        else:
            out = cur_sum * tf.sign(trigger)
        return out, next_state

class ProductCell(rnn_cell_impl.RNNCell):
    def __init__(self,
                name=None):
        super(ProductCell, self).__init__(name=name)
    @property
    def state_size(self):
        return 1
    @property
    def output_size(self):
        return 1
    def build(self, inputs_shape):
        pass
    def call(self, inputs, state):
        out = inputs * state
        return out, out

class ModifiedGRUCell(rnn_cell_impl.RNNCell):
    # GRU Cell modified to use as decoder cell like in bahdanau article
    def __init__(self,
            num_units,
            activation=None,
            reuse=None,
            kernel_initializer=None,
            bias_initializer=None,
            name=None,
            dtype=None):
        super(ModifiedGRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
        
        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)
        
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

        input_depth = inputs_shape[1].value
        self._gate_kernel = self.add_variable(
            "gates/%s" % 'kernel',
            shape=[input_depth + self._num_units * 2, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % 'bias',
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % 'kernel',
            shape=[input_depth + self._num_units * 2, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % 'bias',
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self.built = True

    def call(self, inputs, state_in):
        """Gated recurrent unit (GRU) with nunits cells."""
        state, context_vector = state_in
        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state, context_vector], 1), self._gate_kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._gate_bias)
        
        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        
        r_state = r * state
        
        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state, context_vector], 1), self._candidate_kernel)
        candidate = tf.nn.bias_add(candidate, self._candidate_bias)
        
        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, (new_h, new_h)

def make_bags(sparse,
              embedded,
              splitters,
              embedding_size,
              positional_encoding_bool=False):
    # get parameters
    batch_size = tf.shape(sparse)[0]
    embedded_mask = tf.cast(tf.sign(sparse), tf.float32)
    embedded_len = length_int1(sparse)
    # max_embedded_len = tf.reduce_max(embedded_len)
    # calculate array of triggers
    num_splitters = len(splitters)
    splitters = tf.reshape(splitters, [1, 1, num_splitters])
    so = tf.reshape(sparse, [batch_size, -1, 1])
    max_embedded_len = tf.shape(so)[1]
    triggered = tf.reduce_sum(1 - tf.sign(tf.abs(so - splitters)), axis=2)
    triggered = tf.cast(triggered, tf.float32)
    # make last trigger one
    shifted_mask = tf.concat([embedded_mask[:, 1:], tf.zeros([batch_size, 1])], axis=1)
    triggered = triggered * shifted_mask + (1 - shifted_mask)
    # remove doubled triggers and make first trigger zero
    triggered = triggered \
            * (1 - tf.concat([tf.ones([batch_size, 1]), triggered[:, :-1]], axis=1))
    triggered = tf.concat([triggered[:, :-1], tf.ones([batch_size, 1])], axis=1)
    triggers = tf.reshape(triggered, [batch_size, -1, 1])
    # indices for shorter sequence
    indices0, _ = tf.nn.dynamic_rnn(
        SumWithTriggersCell(1, reset=False),
        tf.concat([triggers, triggers], axis=2),
        dtype=tf.float32
    )
    # positional encoding
    if positional_encoding_bool:
        pe_triggers = tf.concat([tf.ones([batch_size, 1, 1]), triggers[:, 1:, :]], axis=1)
        pe_lengths, _ = tf.nn.dynamic_rnn(
            SumWithTriggersCell(1, reset=True),
            tf.concat([tf.reverse(pe_triggers, [1]),
                    tf.ones([batch_size, max_embedded_len, 1])], axis=2),
            dtype=tf.float32
        )
        pe_lengths = tf.reverse(pe_lengths, [1])
        pe_lengths_copied, _ = tf.nn.dynamic_rnn(
            SumWithTriggersCell(1, reset=True, return_current=True),
            tf.concat([triggers, 
                    pe_lengths], axis=2),
            dtype=tf.float32
        )
        pe_lengths_copied = pe_lengths_copied * (1 - triggers) + pe_lengths
        pe_lengths_reversed = 1 / pe_lengths_copied
        pe_indices, _ = tf.nn.dynamic_rnn(
            SumWithTriggersCell(1, reset=True, return_current=True),
            tf.concat([triggers,
                    pe_lengths_reversed], axis=2),
            dtype=tf.float32
        )
        pe_indices += pe_lengths_reversed
        pe_indices -= triggers * (1 + pe_lengths_reversed)
        tf.identity(pe_lengths)
        bag_input = tf.concat([triggers, pe_indices, embedded], axis=2)
    else:
        bag_input = tf.concat([triggers, embedded], axis=2)
    # make bag of tokens
    bags, _ = tf.nn.dynamic_rnn(
        SumWithTriggersCell(embedding_size,
                positional_encoding=positional_encoding_bool,
                reset=True),
        bag_input,
        dtype=tf.float32
    )
    # project to shorter sequence
    indices = tf.cast(indices0, tf.int32)[:, :, 0]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
    batch_nums = tf.tile(batch_nums, [1, max_embedded_len])
    indices = tf.stack((batch_nums, indices), axis=2)
    print('indices.shape', indices.shape)
    story = tf.scatter_nd(indices, bags, [batch_size, max_embedded_len, embedding_size])
    print('story.shape', story.shape) # (-1, max_embedded_len, embedding_size)
    story = story[:, 1:, :]
    # normalize story
    story_sum = tf.reduce_sum(tf.square(story), 2)
    norm = tf.reshape(tf.sqrt(story_sum), [batch_size, max_embedded_len - 1, 1]) + 1e-8
    # change embedded parameters
    story_sum = tf.sign(story_sum)
    embedded_len = length_int1(story_sum)
    max_embedded_len = tf.reduce_max(embedded_len)
    story = story[:, :max_embedded_len, :] / norm[:, :max_embedded_len, :]
    story_sum = story_sum[:, :max_embedded_len]
    embedded_mask = tf.cast(tf.sign(story_sum), tf.float32)   

    #if hparams['copy']:
    #    raise BaseException('Cannot use copy with bag of tokens')
    tf.shape(story_sum, name='story_shape') # story shape
    
    return story, embedded_mask, embedded_len, max_embedded_len


    

