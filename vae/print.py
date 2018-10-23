import sys
import easyfs
import copy
import numpy as np
import string
import os
import re
from subprocess import check_output
from multiprocessing import Pool

from load_hparams import loader_func
from multiencoder_utils import Multiencoder, NormalizeEncoderSettings


n_lines_to_print = 1000

list_hparams = [NormalizeEncoderSettings(loader_func(f)) for f in sys.argv[1:]]

letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz'

pairs_path = list_hparams[0]['pairs_path']
if len(list_hparams) == 1:
    out_path = list_hparams[0]['answers_path'] + 'visual.html'
    page_name = 'Visual ' + list_hparams[0]['model_name']
else:
    names = '+'.join(h['model_name'] for h in list_hparams[1:])
    out_path = list_hparams[0]['answers_path'] + 'diff_%s.html' % names
    page_name = 'Diff ' + list_hparams[0]['model_name']

# read pairs
test_input_file = pairs_path + 'test_input.txt'
test_output_file = pairs_path + 'test_output.txt'
lines_in = [l.strip() for l in open(test_input_file)]
lines_out = [l.strip() for l in open(test_output_file)]
pairs = [p for p in zip(lines_in, lines_out)]

def conv_line(line):
    prep = line.lower().replace(' ', '').strip()
    prep = re.sub(r'\<[^>]*\>', '', prep)
    for c in string.punctuation:
        prep = prep.replace(c, '')
    return prep

table = {}

def process_hp(hparams):
    # read generated answers
    f = open(hparams['answers_path'] + 'answers.txt')
    answers = [l[:-1] for l in f]
    # read p_gen
    if hparams['copy']:
        f = open(hparams['answers_path'] + 'p_gen.txt')
        p_gens = []
        line = ''.join([l[:-1] for l in f])
        for l in line.split('['):
            p = l[:-1].split(' ')
            p = [float(pp) for pp in p if pp != '']
            p_gens.append(p)
        p_gens = p_gens[1:]
    if len(answers) != len(pairs):
        print('Error: len(answers) != len(pairs) (%d != %d)' % (len(answers), len(pairs)))
    # calculate scores
    correct_answers = sum([1 if conv_line(a) == conv_line(p[1]) else 0 for (a, p) in zip(answers, pairs)])
    bleu_scores = str(check_output('t2t-bleu --translation=%sanswers.txt --reference=%stest_output.txt' % (hparams['answers_path'], list_hparams[0]['pairs_path']), shell=True)).replace('\\n', '\n')[2:-1].strip()
    print(hparams['model_name'])
    print('Accuracy: %.03f' % (correct_answers / len(answers)))
    print(bleu_scores)
    f = open(hparams['answers_path'] + 'scores.txt', 'w')
    f.write('Accuracy: ' + str(correct_answers / len(answers)) + '\n')
    f.write(bleu_scores + '\n')
    # calculate top-n accuracy
    if hparams['use_beam_search']:
        topn_lines = [l.strip() for l in open(hparams['answers_path'] + 'answers_topn.txt')]
        count_top = [0 for _ in range(hparams['beam_width'])]
        for i, pair in enumerate(pairs):
            story, exp_answer = pair
            tn = []
            for j in range(hparams['beam_width']):
                tn.append(topn_lines[i * hparams['beam_width'] + j])
            tn = [conv_line(l.split('\t')[0]) for l in tn]
            if conv_line(exp_answer) in tn:
                for j in range(tn.index(conv_line(exp_answer)), hparams['beam_width']):
                    count_top[j] += 1
        for n, cn in enumerate(count_top):
            f.write(hparams['model_name'] + ' Top-%i accuracy: ' % (n + 1) + str(count_top[n] / len(answers)) + '\n')
    # make copied parts colored blue
    if hparams['copy']:
        encoder = Multiencoder([hparams['output_encoder']], hparams['max_answer_len'])
        new_answers = []
        for i, answer in enumerate(answers):
            subwords = encoder.decode_list(encoder.encode(answer))[0]
            words = []
            for j, w in enumerate(subwords):
                if j < hparams['max_answer_len'] and hparams['copy'] and not ('train_only_vectors' in hparams and hparams['train_only_vectors']):
                    p = p_gens[i][j]
                else:
                    p = 1
                c = '%02X' % int((1 - p) * 255)
                for pun in string.punctuation + '\\u':
                    if j != len(subwords) - 1 and pun != '_' and pun in subwords[j + 1]:
                        w = w.replace('_', '')
                    if pun != '_' and pun in w:
                        rep = True
                        for l in letters:
                            if l in w.lower():
                                rep = False
                        if rep:
                            w = w.replace('_', '')
                w = w.replace('_', ' ')
                w1 = '<font color=#0000%s title="p_gen = %f">' % (c,p) + w + '</font>'
                words.append(w1)
            ans = ''.join(words)
            ans = ans.replace('\\u', '_')
            new_answers.append(ans)
        answers = new_answers
    # write table
    copy_info, beam_info = '', ''
    if hparams['use_beam_search']:
        beam_info = '<h3>Beam size %i</h3>' % hparams['beam_width']
    if hparams['copy']:
        copy_info = '<h3><font title="Hover mouse over words to see generation probability" color=#0000ff >Using copy<font></h3>'
    result = {
        'header': '<h3>Model: ' + hparams['model_name'] + '</h3><br>' \
                + 'Accuracy: %.03f' % (correct_answers / len(answers)) + '<br>' \
                + bleu_scores.replace('\n', '<br>') \
                + beam_info + copy_info,
        'content': answers,
    }
    return result

pool = Pool()
results = pool.map(process_hp, list_hparams)
for r, hparams in zip(results, list_hparams):
    table[hparams['model_name']] = r

# generate html page
page = ''

# html intro
page += '<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <title>%s</title>\n</head>\n' % page_name
page += '<style>html, body{ /*font-family:Tahoma, Geneva, sans-serif;*/ font-family:Arial, Helvetica, sans-serif; padding:5px; margin:5px;}table { border-collapse: collapse; width: 100%; padding: 5px; marging: 5px; font-size: 80%}td, th { border: 1px solid #ddd; text-align: left; padding: 5px; marging: 5px; -ms-word-break: break-all; word-break: break-all; /* Non standard for webkit */ word-break: break-word; -webkit-hyphens: auto; -moz-hyphens: auto; -ms-hyphens: auto; hyphens: auto;}ol { padding-left: 18px; marging: 0px;}li { padding-left: 1px; marging: 0px;}td.correct_0 { background-color:#ffdddd}td.correct_1 { background-color:#dfffdd}</style>'

page += '<body>\n<table style="width:100%; table-layout:fixed;">\n'

# table headers
width_percent = 95 // (len(list_hparams) + 3)
page += '<tr>\n<th style="width:5%%;">ID</th>\n<th style="width:%i%%;">Sentence</th>\n' % width_percent
for hparams in list_hparams:
    page += '<th style="width:%i%%;">%s</th>\n' % (width_percent, table[hparams['model_name']]['header'])
page += '</tr>\n'

# table content
for i in range(len(pairs)):
    page += '<tr>\n'
    page += '<td align="left" style="width:5%%;">%i</td>\n' % i
    # source
    source = pairs[i][0]
    source = source.replace('CLIENT', '|').replace('SYSTEM', '|').replace('API', '|')
    source = [s for s in source.split('|') if s != '']
    source_txt = '<br>'
    if i > n_lines_to_print:
        break
    for j, rep in enumerate(source[:-1]):
        source_txt += '%i. %s<br>' % (j, rep)
    page += '<td align="left" style="width:%i%%;">%s</td>\n' % (width_percent, source[-1])
    #page += '<td align="left" style="width:%i%%;">%s</td>\n' % (width_percent, source_txt)
    # ref answer
    #page += '<td align="left" style="width:%i%%;">%s</td>\n' % (width_percent, pairs[i][1])
    # check if all answers are the same
    same_answers = True
    for hparams in list_hparams[1:]:
        if (conv_line(table[hparams['model_name']]['content'][i]) !=
                conv_line(table[list_hparams[0]['model_name']]['content'][i])):
            same_answers = False
    if len(list_hparams) == 1:
        same_answers = False
    # model answers
    for hparams in list_hparams:
        if conv_line(table[hparams['model_name']]['content'][i]) == conv_line(pairs[i][1]):
            color = '#dfffdd'
        else:
            color = '#ffdddd'
        if same_answers:
            color = color.replace('d', 'e')
        page += '<td align="left" style="width:%i%%; background-color:%s;">%s</td>\n' \
                % (width_percent, color, table[hparams['model_name']]['content'][i])
    page += '</tr>\n'
# ending
page += '</table></body>\n'

# save page
open(out_path, 'w').write(page)
print('\nsaved page as', out_path)



















