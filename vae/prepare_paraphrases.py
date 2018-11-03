import easyfs
import string
import os
import random

from load_hparams import hparams, PrintHparamsInfo

PrintHparamsInfo(hparams)

# Create pairs folder if necessary
if not os.path.exists(hparams['pairs_path']):
    os.makedirs(hparams['pairs_path'])

corpus = []

# get single comments
for fname in easyfs.onlyFiles(hparams['comments_path'], full_path=True):
    print(fname)
    for l in open(fname):
        parts = l.split('\t')
        if len(parts) != 4:
            continue
        text = parts[3]
        text = text.lower()
        # remove reply info from text
        text = text.strip()
        if len(text) == 0:
            continue
        if text[0] == '[' and ']' in text:
            text = text[list(text).index(']'):]
        text = text.strip()
        for _ in range(10):
            for p in string.punctuation:
                if len(text) == 0:
                    break
                if text[0] == p:
                    text = text[1:]
                text = text.strip()
        if len(text) == 0:
            continue
        corpus.append(text)
print('len(corpus)', len(corpus))

# get paraphrased pairs
for fname in easyfs.onlyFiles(hparams['paraphrases_path'], full_path=True):
    print(fname)
    for l in open(fname):
        parts = l.split('\t')
        if len(parts) < 3:
            continue
        text1, text2 = parts[1], parts[2]
        text1 = text1.replace(' ,', ',').replace(' .', '.').lower().strip()
        text2 = text2.replace(' ,', ',').replace(' .', '.').lower().strip()
        corpus.append(text1 + '\t' + text2)
print('len(corpus)', len(corpus))

# shuffle
random.seed(0)
random.shuffle(corpus)

# write output
dev_in = open(hparams['pairs_path'] + 'dev_input.txt', 'w')
dev_out = open(hparams['pairs_path'] + 'dev_output.txt', 'w')
test_in = open(hparams['pairs_path'] + 'test_input.txt', 'w')
test_out = open(hparams['pairs_path'] + 'test_output.txt', 'w')
train_in = open(hparams['pairs_path'] + 'train_input.txt', 'w')
train_out = open(hparams['pairs_path'] + 'train_output.txt', 'w')
n_dev_test = 100
for i, text in enumerate(corpus):
    if i < n_dev_test:
        ifile, ofile = dev_in, dev_out
    elif i < 2 * n_dev_test:
        ifile, ofile = test_in, test_out
    else:
        ifile, ofile = train_in, train_out
    if '\t' in text:
        text1, text2 = text.split('\t')
        if i >= 2 * n_dev_test:
            # normal order
            ifile.write(text1 + '\n')
            ofile.write(text1 + '\n')
            ifile.write(text2 + '\n')
            ofile.write(text2 + '\n')
            # reversed order
            if hparams['include_paraphrases']:
                ifile.write(text1 + '\n')
                ofile.write(text2 + '\n')
                ifile.write(text2 + '\n')
                ofile.write(text1 + '\n')
        else:
            for _ in range(10):
                ifile.write(text1 + '\n')
                ofile.write(text1 + '\n')
            for _ in range(10):
                ifile.write(text2 + '\n')
                ofile.write(text2 + '\n')
    else:
        if i >= 2 * n_dev_test:
            ifile.write(text + '\n')
            ofile.write(text + '\n')
        else:
            for _ in range(10):
                ifile.write(text + '\n')
                ofile.write(text + '\n')
            























