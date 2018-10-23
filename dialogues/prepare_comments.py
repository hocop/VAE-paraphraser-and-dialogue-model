import sys
sys.path.append('../vae')
import easyfs
import numpy as np
import copy
#from gensim.models import Word2Vec
import pickle
import re
import collections
from tensor2tensor.data_generators import text_encoder
import random
from random import shuffle
import os
import string

from load_hparams import hparams

num_test_dev = 1000

path = hparams['text_path']
pairs_path = hparams['pairs_path']

files = [f for f in easyfs.onlyFiles(path, full_path=True)]

# Create pairs folder if necessary
if not os.path.exists(pairs_path):
    os.makedirs(pairs_path)

# pair preprocessing function
def preprocess(request, answer, my_id):
    request = request.lower()
    request = request.replace('tome', 'tOmE')
    answer = answer.lower()
    for _ in range(2):
        to_name = re.search('\[(.+?)\]', request)
        if to_name is not None:
            request = request.replace(to_name.group(0), '')
    # remove reply info from request
    request = request.strip()
    if request[0] == '[' and ']' in request:
        request = request[list(request).index(']'):]
    for _ in range(10):
        for p in string.punctuation:
            if len(request) == 0:
                break
            if request[0] == p:
                request = request[1:]
            request = request.strip()
    # answer
    to_id = re.match('\[(.+?)\]', answer)
    # remove reply info from answer
    answer = answer.strip()
    if answer[0] == '[' and ']' in answer:
        answer = answer[list(answer).index(']'):]
    for _ in range(10):
        for p in string.punctuation:
            if len(answer) == 0:
                break
            if answer[0] == p:
                answer = answer[1:]
            answer = answer.strip()
    return request, answer

all_pairs = []
for fname in files:
    print(fname)
    c_start = len(all_pairs)
    f = open(fname)
    comments_dict = {}
    # read file line by line
    for l in f:
        parts = l.strip().split('\t')
        if len(parts) != 4:
            continue
        com_id, from_id, to_com_id, text = parts
        if len(text) < 2 or max([len(t) for t in text.split(' ')]) > 25 or len(text.split(' ')) > 70:
            continue
        # find responses
        if to_com_id != 'None' and to_com_id in comments_dict:
            pair = preprocess(' tOmE '.join(comments_dict[to_com_id]), text, from_id)
            all_pairs.append(pair)
            comments_dict[com_id] = comments_dict[to_com_id][1], text
        else:
            comments_dict[com_id] = '', text
    print('found', len(all_pairs) - c_start)

print('Total found', len(all_pairs))

# split to three sets
random.seed(0)
shuffle(all_pairs)
train = all_pairs[num_test_dev * 2:]
dev = all_pairs[num_test_dev: num_test_dev * 2]
test = all_pairs[:num_test_dev]

# write pairs
for name, pairs in [('train', train), ('dev', dev), ('test', test)]:
    out_so = open(pairs_path + name + '_input.txt', 'w')
    out_a = open(pairs_path + name + '_output.txt', 'w')
    for p in pairs:
        out_so.write(p[0] + '\n')
        out_a.write(p[1] + '\n')













