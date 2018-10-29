import numpy as np
import tensorflow as tf
import time
import sys
import os
from dialog_model import get_model_fn

sys.path.append('../vae')
from load_hparams import hparams
import easyfs

# Set up logging
tensors_to_log = {'loss': 'loss'}
logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

# Create the Estimator
model_fn = get_model_fn(hparams)
chkp_config = tf.estimator.RunConfig(keep_checkpoint_max=hparams['keep_checkpoint_max'])
estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=hparams['checkpoints_path'],
        config=chkp_config)

# so: source, a: answer
chunk_so_files = sorted([fname for fname in easyfs.onlyFiles(hparams['data_path']) if 'so_train' in fname])
chunk_a_files = sorted([fname for fname in easyfs.onlyFiles(hparams['data_path']) if 'a_train' in fname])

so_eval = np.load(hparams['data_path'] + 'so_dev0.npy')
a_eval = np.load(hparams['data_path'] + 'a_dev0.npy')

for epoch_i in range(hparams['num_epochs']):
    for so_fname, a_fname in zip(chunk_so_files, chunk_a_files):
        # Train one chunk
        start = time.time()
        so_data = np.load(hparams['data_path'] + so_fname)
        a_data = np.load(hparams['data_path'] + a_fname)
        print('Chunk:', so_fname, so_data.shape, a_fname)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'mu_sigma': so_data},
                y=a_data,
                batch_size=hparams['batch_size_train'],
                num_epochs=10,
                shuffle=True)
        estimator.train(
                input_fn=train_input_fn,
                hooks=[logging_hook])
        print('Chunk training took', time.time() - start, 'sec')
        
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"mu_sigma": so_eval},
                y=a_eval,
                num_epochs=1,
                shuffle=False)
        eval_results = estimator.evaluate(input_fn=eval_input_fn)
        print(eval_results)
