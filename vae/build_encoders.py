import easyfs
import numpy as np
import copy
from tensor2tensor.data_generators import text_encoder, tokenizer
import sys
import os
from load_hparams import hparams, PrintHparamsInfo
from collections import Counter
from shutil import copyfile
from multiencoder_utils import Multiencoder, NormalizeEncoderSettings
import string
 
#def is_float(tok): # (rb) this function does not work for strings like '10kg'
#    try:
#        float(tok)
#        return True
#    except:
#        return False

def has_digits(s):
    for ss in s:
        if ss in string.digits:
            return True
    return False

hparams = NormalizeEncoderSettings(hparams)

PrintHparamsInfo(hparams)

pairs_path = hparams['pairs_path']

train_input_file = pairs_path + 'train_input.txt'
train_output_file = pairs_path + 'train_output.txt'

# read pairs
lines_in = [l.strip() for i, l in enumerate(open(train_input_file, 'r', encoding='utf-8')) if i < hparams['max_lines_for_encoder']]
lines_out = [l.strip() for i, l in enumerate(open(train_output_file, 'r', encoding='utf-8')) if i < hparams['max_lines_for_encoder']]
all_pairs = [p for p in zip(lines_in, lines_out)]

corpus = [p[0] for p in all_pairs]
corpus += [p[1] for p in all_pairs]

input_encoders = list(hparams['input_encoders'])
output_encoder = hparams['output_encoder']
all_encoders = list(input_encoders) + [output_encoder]


def BuildCharList(target_size, drop_ends = False):
    encoder = text_encoder.SubwordTextEncoder.build_from_generator(generator = corpus, 
                                                                   target_vocab_size = target_size/2, 
                                                                   max_subtoken_length = 1)
    char_list = encoder._all_subtoken_strings[2:]
    char_list_ends = [s + '_' for s in char_list]
    if drop_ends:
        char_list_full = char_list 
    else:
        char_list_full = char_list + char_list_ends 
    char_list_full.extend([str(d) for d in range(10) if str(d) not in char_list_full])
    char_list_full.extend([str(d) + '_' for d in range(10) if str(d)+'_' not in char_list_full])
    return char_list_full + ['\\u', '\\u_']

def BuildStatFromCorpus(callback, filter_freq = -1, input_delimiters=[], delim_ends=''):
    if filter_freq < 0:
        filter_freq = 0
    ignore_delimiters = len(input_delimiters) > 0

    token_dict = [{}, {}]
    size = len(all_pairs)
    for idx in range(size): 
        p = all_pairs[idx]
        for i in range(2):
            if i == 0 and not ignore_delimiters:
                last_delimiter_idx = max([0] + [p[i].rfind(delimiter) for delimiter in input_delimiters])
                if last_delimiter_idx > len(p[i])-2:
                    continue
                string_to_be_tokenized = p[i][last_delimiter_idx+2 if last_delimiter_idx > 0 else 0:]
            else:
                string_to_be_tokenized = p[i]
            toks = tokenizer.encode (string_to_be_tokenized)
            toks_processed = []
            for tok in toks:
                #if is_float(tok):
                #    continue
                if has_digits(tok):
                    continue
                toks_processed.extend(callback(tok.strip()))
            for tok in toks_processed:
                if tok in token_dict[i]:
                    token_dict[i][tok] += 1
                else:
                    token_dict[i][tok] = 1     
        if idx % 1000 == 0: 
            print('    %d%%' % int(idx / size*100), end='\r')
    token_dict = dict(Counter(token_dict[0])+Counter(token_dict[1]))
    del_keys = []
    for key in token_dict.keys():
        if token_dict[key] <= filter_freq:
            del_keys.append(key)
    for key in del_keys:
            token_dict.pop(key)
    digits = []
    digits.extend([str(d) for d in range(10) if str(d) not in digits])
    digits.extend([str(d) + '_' for d in range(10) if str(d)+'_' not in digits])
    digits.append('.')
    return digits + list(token_dict.keys())
    
def BuildBigramList(filter_freq, drop_ends = False):
    input_delimiters = ''
    if 'input_delimiters' in hparams:
        input_delimiters = [delim + '_' for delim in hparams['input_delimiters']]
    return ['UNK'] + BuildStatFromCorpus(lambda x: \
                         Multiencoder.prepare_ngrams(x, 2, drop_ends), \
                         filter_freq, \
                         input_delimiters) + input_delimiters

def BuildTrigramList(filter_freq, drop_ends = False):
    input_delimiters = ''
    if 'input_delimiters' in hparams:
        input_delimiters = [delim + '_' for delim in hparams['input_delimiters']]
    return ['UNK'] + BuildStatFromCorpus(lambda x: \
                         Multiencoder.prepare_ngrams(x, 3, drop_ends), \
                         filter_freq, \
                         input_delimiters) + input_delimiters
def BuildTokenList(filter_freq, drop_ends = False):
    input_delimiters = ''
    if 'input_delimiters' in hparams:
        input_delimiters = hparams['input_delimiters']
    return ['UNK'] + \
           BuildStatFromCorpus(lambda x:[Multiencoder.prepare_token(x)], filter_freq) + \
           input_delimiters

def GetCleanedWordpieceList(encoder): 
    #wordpieces = [s for s in encoder._all_subtoken_strings[2:] if not is_float(s[:-1]) and not is_float(s)]
    wordpieces = [s for s in encoder._all_subtoken_strings[2:] if not has_digits(s)] # (rb) added
    wordpieces.extend([str(d) for d in range(10) if str(d) not in wordpieces])
    wordpieces.extend([str(d) + '_' for d in range(10) if str(d)+'_' not in wordpieces])
    #wordpieces.extend([d + '_' for d in wordpieces if d.isalnum() and d + '_' not in wordpieces])
    
    return wordpieces + ['\\u', '\\u_']

def CleanWordpieceEncoderNumbers(encoder, add_end_token=False): # (rb) unused function?
    wordpiece_encoder = text_encoder.SubwordTextEncoder()
    wordpieces = GetCleanedWordpieceList(encoder)
    if add_end_token:
        wordpieces.append('N_')
    wordpieces = ['<pad>', '<EOS>'] + wordpieces
    wordpiece_encoder._init_subtokens_from_list(wordpieces)
    wordpiece_encoder._init_alphabet_from_tokens(wordpieces)
    return wordpiece_encoder
    
def BuildWordpieceList(target_size, drop_ends=False):
    encoder = text_encoder.SubwordTextEncoder.build_from_generator(
            corpus,
            target_size)
    wordpieces = GetCleanedWordpieceList(encoder)
    return wordpieces

call_dict = {'char': BuildCharList, 
             '2char': BuildBigramList, 
             '3char': BuildTrigramList, 
             'wordpiece': BuildWordpieceList, 
             'token': BuildTokenList}
encoders_to_build = []
for enc in all_encoders:
    if enc['build']:
        encoders_to_build.append(enc)

encoders_to_build = [] # (rb) repeated code?
for enc in all_encoders:
    if enc['build']:
        encoders_to_build.append(enc)

call_dict = {'char': BuildCharList, 
             'bigram': BuildBigramList, 
             'trigram': BuildTrigramList, 
             'wordpiece': BuildWordpieceList, 
             'token': BuildTokenList}

for enc in encoders_to_build:
    size_param = 0
    if 'target_size' in enc:
        size_param = enc['target_size']
    elif 'frequency_cut' in enc:
        size_param = enc['frequency_cut']
    print('Building encoder %s, size_param = %d' % (enc['type'], size_param))
    reserved_token_list = ['<pad>', '<EOS>', 'NEWWORD']
    if 'tokenizer_emit_spaces' in hparams and hparams['tokenizer_emit_spaces']:
        reserved_token_list.append('П') # used as space symbol to keep it after all the strips
    if len(hparams['input_encoders']) > 1 or hparams['input_encoders'][0]['bag']:
        reserved_token_list.append('К') # used as end of number
    token_list = reserved_token_list + call_dict[enc['type']](size_param, enc['drop_end'])
    #token_list = call_dict[enc['type']](size_param, enc['drop_ends'])
    print('size = %d\n' % len(token_list))
    with open (enc['path'], 'w') as enc_file:
        for t in token_list:
            enc_file.write("'%s'\n" % t)








