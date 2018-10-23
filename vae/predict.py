import numpy as np
import tensorflow as tf
import pickle
import easyfs
import copy
import re
import string
import sys
import os

from my_model import get_model_fn

tf.logging.set_verbosity(tf.logging.INFO)

from load_hparams import hparams, PrintHparamsInfo
from multiencoder_utils import Multiencoder, NormalizeEncoderSettings
from generator_utils import *

hparams = NormalizeEncoderSettings(hparams)

PrintHparamsInfo(hparams)

# Create answers folder if necessary
if not os.path.exists(hparams['answers_path']):
    os.makedirs(hparams['answers_path'])

if ('draw_attention' in hparams and hparams['draw_attention']) or ('draw_entropy' in hparams and hparams['draw_entropy']):
    import matplotlib.pyplot as plt

model_fn = get_model_fn(hparams)

encoder = Multiencoder([hparams['output_encoder']], hparams['max_answer_len'])

def main(unused_argv):
    # Create the Estimator
    estimator = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=hparams["checkpoints_path"])
    
    # read sources
    if ('draw_attention' in hparams and hparams['draw_attention']) or ('draw_entropy' in hparams and hparams['draw_entropy']):
        sources = [l.strip() for l in open(hparams['pairs_path'] + 'test_input.txt')]
        in_encoder = Multiencoder(hparams['input_encoders'], hparams['max_answer_len'])
        from tensor2tensor.data_generators import text_encoder, tokenizer
    
    # Predict function
    so_file = open(hparams['pairs_path'] + 'test_input.txt')
    predict_input_fn = lambda: i_input_fn(so_file, hparams)
    
    # Prepare for making predictions
    results = estimator.predict(input_fn=predict_input_fn)
    if not os.path.isdir(hparams["answers_path"]):
        os.system('mkdir ' + hparams["answers_path"])
    f1 = open(hparams['answers_path'] + 'answers.txt', 'w')
    if not ('train_only_vectors' in hparams and hparams['train_only_vectors']) and hparams['use_beam_search']:
        f1_topn = open(hparams['answers_path'] + 'answers_topn.txt', 'w')
    if hparams['copy']:
        f2 = open(hparams['answers_path'] + 'p_gen.txt', 'w')
    entropy = np.zeros(hparams['max_answer_len'])
    count = np.zeros(hparams['max_answer_len'])
    
    # Get predictions
    for i, r in enumerate(results):
        sent_vec = r['classes']
        if 1 in sent_vec:
            sent_vec = sent_vec[:list(sent_vec).index(1)]
        sent = encoder.decode(sent_vec)
        f1.write(sent + '\n')
        if hparams['use_beam_search']:
            for j in range(hparams['beam_width']):
                sent_j = r['classes_topn'][:, j]
                if 1 in sent_j:
                    sent_j = sent_j[:list(sent_j).index(1)]
                sent_j_w = encoder.decode(sent_j)
                f1_topn.write(sent_j_w + '\t' + str(r['beam_scores'][0, j]) + '\n')
        if hparams['copy']:
            f2.write(str(r['p_gens']) + '\n')
        # draw attention matrix image
        if ('draw_attention' in hparams and hparams['draw_attention']) or ('draw_entropy' in hparams and hparams['draw_entropy']):
            img = r['attention_image']
            print(sources[i])
            print(in_encoder.decode_list(in_encoder.encode(sources[i])))
            in_words = []
            cur_words = []
            for w in in_encoder.decode_list(in_encoder.encode(sources[i]))[0]:
                if 'NEWWORD' in w:
                    in_words.append(' '.join(cur_words).replace('_', '') + '  ')
                    cur_words = []
                else:
                    cur_words.append(w)
            out_words = encoder.decode_list(encoder.encode(sent))[0]
            img = img[:len(in_words), :len(sent_vec)]
        if 'draw_attention' in hparams and hparams['draw_attention']:
            plt.figure(figsize=(len(out_words), len(in_words)))
            plt.imshow(img, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
            plt.yticks(range(len(in_words)), in_words)
            plt.xticks(range(len(out_words)),
                    [w for j, w in enumerate(out_words)],
                    rotation='vertical')
            mng = plt.get_current_fig_manager()
            plt.show()
        # calc entropy
        if 'draw_entropy' in hparams and hparams['draw_entropy']:
            ent = np.zeros(hparams['max_answer_len'])
            ent[:len(sent_vec)] = -np.sum(img * np.log(img + 1e-8), axis=0).reshape(len(sent_vec))
            entropy += ent / np.log(len(in_words) + 2)
            mask = np.zeros(hparams['max_answer_len'])
            mask[:len(sent_vec)] = 1
            count += mask
    entropy /= count + 1e-8
    if 'draw_entropy' in hparams and hparams['draw_entropy']:
        plt.title('Entropy of attention distribution')
        plt.xlabel('Decoder step')
        plt.ylabel('Relative entropy')
        plt.plot(entropy)
        plt.show()

if __name__ == "__main__":
    tf.app.run()










