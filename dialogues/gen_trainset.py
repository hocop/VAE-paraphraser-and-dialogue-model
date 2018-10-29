import os
import sys
import numpy as np

sys.path.append('../vae')
from query_paraphrases import encode_batch
from load_hparams import hparams

# Create data folder if necessary
if not os.path.exists(hparams['data_path']):
    os.makedirs(hparams['data_path'])

# Read trainset
for fname in ['dev', 'train']:
    chunk_so = []
    chunk_a = []
    batch_so = []
    batch_a = []
    chunks_count = 0
    # so: source, a: answer
    for so, a in zip(open(hparams['pairs_path'] + fname + '_input.txt'),
                        open(hparams['pairs_path'] + fname + '_output.txt')):
        so = so.strip().lower()
        a = a.strip().lower()
        if len(so) == 0 or len(a) == 0:
            continue
        batch_so.append(so)
        batch_a.append(a)
        if len(batch_so) == hparams['batch_size']:
            try:
                b_so = [so for so in encode_batch(batch_so)]
                b_a = [a for a in encode_batch(batch_a)]
                chunk_so.extend(b_so)
                chunk_a.extend(b_a)
            except:
                print('skipped batch')
            batch_so = []
            batch_a = []
            if len(chunk_so) >= hparams['chunk_size']:
                chunk_so = np.array(chunk_so).astype('float16')
                chunk_a = np.array(chunk_a).astype('float16')
                np.save(hparams['data_path'] + 'so_' + fname + str(chunks_count) + '.npy', chunk_so)
                np.save(hparams['data_path'] + 'a_' + fname + str(chunks_count) + '.npy', chunk_a)
                chunk_so = []
                chunk_a = []
                print('chunk %i saved' % chunks_count)
                chunks_count += 1
    if len(chunk_so) >= 0:
        chunk_so = np.array(chunk_so).astype('float16')
        chunk_a = np.array(chunk_a).astype('float16')
        np.save(hparams['data_path'] + 'so_' + fname + str(chunks_count) + '.npy', chunk_so)
        np.save(hparams['data_path'] + 'a_' + fname + str(chunks_count) + '.npy', chunk_a)
        print('chunk %i saved' % chunks_count)
