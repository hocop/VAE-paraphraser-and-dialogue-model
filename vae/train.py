import numpy as np
import tensorflow as tf
import time
import sys
import os
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import bleu_hook
from my_model import get_model_fn
from generator_utils import *

import os.path
tf.logging.set_verbosity(tf.logging.INFO)

from load_hparams import hparams, PrintHparamsInfo
from multiencoder_utils import Multiencoder, NormalizeEncoderSettings

# For future versions with distribution
#from tensorflow.python.client import device_lib
#
#def get_available_gpus():
#    local_device_protos = device_lib.list_local_devices()
#    return [x.name for x in local_device_protos if x.device_type == 'GPU']

hparams = NormalizeEncoderSettings(hparams)
PrintHparamsInfo(hparams)

batch_size = hparams['batch_size']
num_epochs = hparams['num_epochs']

# Get current epoch
epoch_data_filename = hparams['checkpoints_path'] + 'current_epoch.txt'
current_epoch = 0
try:
    with open(epoch_data_filename, 'r') as epoch_data_file:
        current_epoch = int(epoch_data_file.read())
except:
    pass
if current_epoch >= num_epochs:
    print('Training is already finished. hparams["num_epochs"] = %i' % hparams['num_epochs'])
    exit()

# Load encoder
output_encoder = Multiencoder([hparams['output_encoder']], hparams['max_answer_len'])

# get max BLEU
try:
    max_bleu_epoch, max_bleu = [float(l.strip()) for l in open(
                hparams['checkpoints_path'] + 'max_bleu.txt')]
except:
    max_bleu_epoch, max_bleu = -1, 0

# Make checkpointing config
chkp_config = tf.estimator.RunConfig(keep_checkpoint_max=hparams['keep_checkpoint_max'])
summary_writer = tf.summary.FileWriter(hparams["checkpoints_path"])
# For future versions with distribution
#distr_strat = tf.contrib.distribute.MirroredStrategy() if len(get_available_gpus()) > 1 else None
#chkp_config = tf.estimator.RunConfig(keep_checkpoint_max=hparams['keep_checkpoint_max'], train_distribute=distr_strat)

# Set up logging
tensors_to_log = {'loss': 'loss', 'in': 'stripped_shape', 'KL': 'loss_kl'}

logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

# maybe read input data
if 'epoch_size' in hparams and hparams['epoch_size'] != 'default':
    so_all = [l.strip() for l in open(hparams['pairs_path'] + 'train_input.txt')]
    a_all = [l.strip() for l in open(hparams['pairs_path'] + 'train_output.txt')]

# Train the model
for epoch_i in range(current_epoch, num_epochs):
    print('Epoch', epoch_i)

    # maybe turn off adam
    if 'disable_adam_after_epoch' in hparams and epoch_i > hparams['disable_adam_after_epoch']:
        hparams['adam'] = False
        hparams['learning_rate'] = hparams['learning_rate_after_disabling_adam']
        print('NOTE: Adam is turned off')
    # maybe turn on decoder speedup
    if 'speedup_after_epoch' in hparams:
        if epoch_i <= hparams['speedup_after_epoch']:
            speedup_backup = hparams['decoder_speedup']
            hparams['decoder_speedup'] = 1
        else:
            if 'speedup_backup' in locals():
                hparams['decoder_speedup'] = speedup_backup
            print('NOTE: Decoder speedup x%i is turned on' % hparams['decoder_speedup'])
    model_fn = get_model_fn(hparams)
    
    # Create the Estimator
    model_dir = hparams['checkpoints_path']
    estimator = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=model_dir,
            config=chkp_config)
    
    # Train one epoch
    start = time.time()
    if 'epoch_size' in hparams and hparams['epoch_size'] != 'default':
        start_point = (epoch_i * hparams['epoch_size']) % len(so_all)
        so_file = so_all[start_point: start_point + hparams['epoch_size']]
        a_file = a_all[start_point: start_point + hparams['epoch_size']]
    else:
        so_file = open(hparams['pairs_path'] + 'train_input.txt')
        a_file = open(hparams['pairs_path'] + 'train_output.txt')
    estimator.train(
            input_fn=lambda: io_input_fn(so_file, a_file, hparams),
            hooks=[logging_hook])
    print('Epoch took', time.time() - start, 'sec')
    
    # Save number of epoch
    epoch_storage_file = open(hparams['checkpoints_path'] + 'current_epoch.txt', 'w')
    epoch_storage_file.write('%d' % (epoch_i + 1))
    epoch_storage_file.close()
    
    # Evaluate the model and print results
    so_file = open(hparams['pairs_path'] + 'dev_input.txt')
    a_file = open(hparams['pairs_path'] + 'dev_output.txt')
    eval_results = estimator.evaluate(input_fn=lambda: io_input_fn(so_file, a_file, hparams))
    # calculate BLEU
    if 'log_bleu_every' in hparams and hparams['log_bleu_every'] and \
                epoch_i % hparams['log_bleu_every'] == 0:
        start = time.time()
        so_file = open(hparams['pairs_path'] + 'dev_input.txt')
        results_prediction = estimator.predict(input_fn=lambda: i_input_fn(so_file, hparams))
        pred_tmp_file = open(hparams['checkpoints_path'] + \
                    ('dev_prediction_%i.txt' % epoch_i), 'w')
        for j, r in enumerate(results_prediction):
            if j % 1000 == 0:
                print('Predicting...', j)
            sent_vec = r['classes']
            if 1 in sent_vec:
                sent_vec = sent_vec[:list(sent_vec).index(1)]
            sent = output_encoder.decode(sent_vec)
            pred_tmp_file.write("%s\n" % sent)
        pred_tmp_file.close()
        print('Predictions took', time.time() - start, 'sec')
        bleu = bleu_hook.bleu_wrapper(ref_filename=hparams['pairs_path'] + 'dev_output.txt', 
                  hyp_filename=hparams['checkpoints_path'] + ('dev_prediction_%i.txt' % epoch_i)) * 100
        print('\033[32;1mBLEU = %f\033[0m' % bleu)
        summary = tf.Summary(value=[tf.Summary.Value(tag='BLEU_predictions', simple_value=bleu)])
        summary_writer.add_summary(summary, epoch_i)
        summary_writer.flush()

print(eval_results)
print('Training finished')


