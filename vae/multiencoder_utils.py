from tensor2tensor.data_generators import text_encoder, tokenizer
import numpy as np
import re
import string
import os.path

def is_float(tok):
    try:
        float(tok)
        return True
    except:
        return False

def divide_numbers(toks):
    res = []
    for tok in toks:
        cur_line = ''
        for c in tok:
            if c in string.digits:
                if len(cur_line) > 0:
                    res.append(cur_line)
                res.append(c)
                cur_line = ''
            else:
                cur_line += c
        if len(cur_line) > 0:
            res.append(cur_line)
    return res

class Multiencoder():
    def __init__(self, encoders, max_source_len, emit_spaces = False, omit_ending_space = False):
        if type(encoders) == dict:
            self.encoders = [encoders]
        else:
            self.encoders = encoders 
        self.max_source_len = max_source_len
        self.t2t_encs = []
        self.newword_codes = []
        self.vocab_max_size = 0
        self.vocab_sizes = []
        self.emit_spaces = emit_spaces
        self.omit_ending_space = omit_ending_space
        self.emit_newword = len(self.encoders) > 1 or self.encoders[0]['bag']
        for enc_idx in range(len(self.encoders)):
            enc = self.encoders[enc_idx]
            vocab = []
            with open(enc['path']) as vocab_file:
                file_size = 0
                for l in vocab_file:
                    file_size += 1
                    l = l.strip()
                    if ((l.startswith("'") and l.endswith("'")) or
                                (l.startswith("\"") and l.endswith("\""))):
                        l = l[1:-1]
                    vocab.append(l)
                    if 'max_vocab_size' in enc and file_size >= enc['max_vocab_size']:
                        break
                if file_size > self.vocab_max_size:
                    self.vocab_max_size = file_size
                self.vocab_sizes.append(file_size)
            enc = self.encoders[enc_idx]
            if 'NEWWORD' in vocab and self.emit_newword:
                self.newword_codes.append(vocab.index('NEWWORD'))
            else:
                self.newword_codes.append(0)
            ste_enc = False
            oov = None

            if enc['type'] == 'copy':
                enc = enc['mirror']

            if enc['type'] == 'char' and enc['drop_end']: 
                ste_enc = False
            elif enc['type'] in ['token', 'bigram', 'trigram']:
                oov = 'UNK' 
                ste_enc = False
            elif (enc['type'] == 'char' and not enc['drop_end']) or \
                 enc['type'] == 'wordpiece':
                ste_enc = True
            if ste_enc:
                self.t2t_encs.append(text_encoder.SubwordTextEncoder())
                self.t2t_encs[enc_idx]._init_subtokens_from_list(vocab)
                self.t2t_encs[enc_idx]._init_alphabet_from_tokens(vocab)
            else:
                self.t2t_encs.append(text_encoder.TokenTextEncoder(None, 
                                                                   vocab_list = vocab, 
                                                                   replace_oov = oov))
    @property
    def vocab_size(self):
        return self.vocab_max_size
    def encode(self, sentence, emit_eos=False):
        toks = tokenizer.encode(sentence)
        if len(toks) == 0:
            print("|%s|" % sentence)
        if self.emit_spaces: 
            toks_spaced = []
            for tok_idx in range(len(toks)):
                toks_spaced.append(toks[tok_idx])
                if not(tok_idx == len(toks) - 1 and self.omit_ending_space):
                    toks_spaced.append('П')
            toks = toks_spaced
        if self.emit_newword:
            toks = divide_numbers(toks)
        results = []
        emitted_nw = 0
        emitted_toks = 0
        #print("Encoding '%s'" % sentence)
        for enc_idx in range(len(self.encoders)):
            results.append([])
            emitted = 0
            enc = self.encoders[enc_idx]

            cut_single_for_copy = False
            if enc['type'] == 'copy':
                cut_single_for_copy = len(self.encoders) > 1
                enc = enc['mirror']

            if enc['type'] == 'char':
                if enc['drop_end']:
                    encode_func = lambda tok: \
                        self.t2t_encs[enc_idx].encode(Multiencoder.prepare_char_drop_end(tok))
                else:
                    encode_func = lambda tok: self.t2t_encs[enc_idx].encode_without_tokenizing(tok)
            elif enc['type'] == 'bigram':
                encode_func = \
                    lambda tok: self.t2t_encs[enc_idx]\
                                    .encode(\
                                        ' '.join(Multiencoder.prepare_ngrams(tok.strip(), 2, enc['drop_end']))
                                           )
            elif enc['type'] == 'trigram':
                encode_func = \
                    lambda tok: self.t2t_encs[enc_idx]\
                                    .encode(\
                                        ' '.join(Multiencoder.prepare_ngrams(tok.strip(), 3, enc['drop_end']))
                                           )
            elif enc['type'] == 'wordpiece':
                encode_func = lambda tok: self.t2t_encs[enc_idx].encode(tok)
            elif enc['type'] == 'token':
                encode_func = lambda tok: self.t2t_encs[enc_idx].encode(self.prepare_token(tok))
            for tok in toks:
                encoded_tok = encode_func(tok)
                if cut_single_for_copy:
                    results[enc_idx].append(encoded_tok[0])
                else:
                    results[enc_idx].extend(encoded_tok)
                if self.emit_newword and enc['type'] != 'token' and not cut_single_for_copy:
                    results[enc_idx].append(self.newword_codes[enc_idx])
                    emitted_nw += 1
                elif enc['type'] == 'token' or cut_single_for_copy:
                    emitted_toks += 1
            if emit_eos:
                results[enc_idx].append(1)
        if emitted_toks % len(toks) or emitted_nw % len(toks):
            raise NameError('Bad number of tokens or NEWWORDs : emitted_toks = %d, emitted_nw = %d, toks = %d' % (emitted_toks, emitted_nw, len(toks))) 
        result = np.zeros([len(self.encoders) * self.max_source_len], dtype=int)
        for enc_idx in range(len(self.encoders)):
            if len(results[enc_idx]) > self.max_source_len:
                raise NameError('Cannot fit sentence in max_*_len = %d slots' % self.max_source_len)
           
        for enc_idx in range(len(self.encoders)):
            sublist_start = self.max_source_len * enc_idx
            result[sublist_start : 
                   sublist_start + 
                   len(results[enc_idx])] = \
                   results[enc_idx]
            #print('Encoding %ss: ' % (self.encoders[enc_idx]['type']), results[enc_idx])
        return result
    
    # count how many units will be when string will be encoded
    def count_length(self, sentence):
        toks = tokenizer.encode(sentence)
        toks = divide_numbers(toks)
        return len(toks)
    
    @staticmethod
    def prepare_char_drop_end(tok):
        if tok == ' ':
            return tok + ' '
        return ' '.join(tok)
    
    @staticmethod
    def prepare_ngrams(tok, grams, drop_end):
        if is_float(tok) or tok == '.':
            return divide_numbers([tok])
        end = grams
        if not drop_end:
            tok_processed = 'Й' + tok + 'Й'
        else:
            tok_processed = tok
        ngrams_list = []
        if end > len(tok_processed):
            ngrams_list = [tok]
        else:
            while end <= len(tok_processed):
                ngrams_list.append(tok_processed[end - grams : end])
                end += 1
        return ngrams_list

    @staticmethod
    def prepare_token(tok):
        if is_float(tok) or tok == '.':
            return ' '.join(divide_numbers([tok]))
        result = re.sub('[' + string.punctuation + ']', '', tok.strip())
        if not len(result):
            return tok
        else:
            return result
       
    def decode(self, data):
        if len(self.encoders) != 1:
            raise NameError("Cannot decode with multiple encoders")
        return self.t2t_encs[0].decode(data)

    def decode_list(self, data):
        results = []
        for enc_idx in range(len(self.encoders)):
            sublist_start = self.max_source_len * enc_idx
            processed = []
            for slot in range(sublist_start, sublist_start + self.max_source_len):
                if data[slot]:
                    processed.append(data[slot])
            results.append(self.t2t_encs[enc_idx].decode_list(processed))
        return results
            

def NormalizeEncoderSettings(hparams):
    input_encoders = hparams['input_encoders']
    common_parameters = ['type', 'path', 'build', 'use_prebuilt_embeddings', 'prebuilt_embeddings_path']
    baggable = ['char', 'bigram', 'trigram'] # (rb) r'nt wordpieces baggable?
    drop_endable = ['char', 'bigram', 'trigram']
    size_parameters = {'char':      ['target_size'],
                       'bigram':    ['frequency_cut'],
                       'trigram':   ['frequency_cut'],
                       'wordpiece': ['target_size'],
                       'token':     ['frequency_cut', 'max_vocab_size']}
    selectable_parameters = dict(size_parameters)
    
    class EncoderDefaults:
        target_size = 500
        frequency_cut = 5
        drop_end = False
    
    for typ in baggable:
        selectable_parameters[typ].extend(['bag', 'positional_encoding'])
    for typ in drop_endable:
        selectable_parameters[typ].extend(['drop_end'])
    
    output_encoder = hparams['output_encoder']
    if output_encoder['type'] not in ['char', 'wordpiece', 'token']:
        raise NameError("Invalid output encoder type: %s" % output_encoder['type'])
    
    if 'target_size' in size_parameters[output_encoder['type']]:
        if 'target_size' not in output_encoder:
            output_encoder['target_size'] = EncoderDefaults.target_size
        size_parameter = output_encoder['target_size']
        size_parameter_suff = 'ts'
    elif 'frequency_cut' in selectable_parameters[output_encoder['type']]:
        if 'frequency_cut' not in output_encoder:
            input_encoder['frequency_cut'] = EncoderDefaults.frequency_cut
        size_parameter = output_encoder['frequency_cut']
        size_parameter_suff = 'fc'
    if 'path' not in output_encoder:
        output_encoder['path'] = hparams['data_path'] + 'encoder_out_' + \
                                 output_encoder['type'] + '_' + \
                                 str(size_parameter) + size_parameter_suff + '.dat'
    if 'build' not in output_encoder:
        output_encoder['build'] = os.path.isfile(output_encoder['path'])
    if 'drop_end' not in output_encoder:
        output_encoder['drop_end'] = EncoderDefaults.drop_end
    output_encoder['bag'] = False 
    copy_encoder_exists = False 
    for input_encoder_idx in range(len(input_encoders)):
        input_encoder = input_encoders[input_encoder_idx]
        if input_encoder['type'] == 'copy':
            if len(input_encoder) > 1:
                raise NameError('Cannot have user-defined settings for "copy" encoder')
            copy_encoder_exists = True
            input_encoder['build'] = False
            input_encoder['mirror'] = output_encoder
            input_encoder['path'] = output_encoder['path'] 
            input_encoder['bag'] = False
            #input_encoder['use_prebuilt_embeddings'] = output_encoder['use_prebuilt_embeddings']
            #input_encoder['prebuilt_embeddings_path'] = output_encoder['prebuilt_embeddings_path']
        else:
            if input_encoder['type'] not in ['char',
                                             'bigram',
                                             'trigram', 
                                             'wordpiece', 
                                             'token']:
                raise NameError("Invalid input encoder: %s" % input_encoder)
            for param in input_encoder:
                if param not in common_parameters and \
                   param not in selectable_parameters[input_encoder['type']]:
                    raise NameError("Cannot use '%s' encoder with parameter '%s': " % \
                                    (input_encoder['type'], param), input_encoder)
            size_parameter = 0
            size_parameter_suff = ''
            if 'target_size' in selectable_parameters[input_encoder['type']]:
                if 'target_size' not in input_encoder:
                    input_encoder['target_size'] = EncoderDefaults.target_size
                size_parameter = input_encoder['target_size']
                size_parameter_suff = 'ts'
            elif 'frequency_cut' in selectable_parameters[input_encoder['type']]:
                if 'frequency_cut' not in input_encoder:
                    input_encoder['frequency_cut'] = EncoderDefaults.frequency_cut
                size_parameter = input_encoder['frequency_cut']
                size_parameter_suff = 'fc'
            if 'path' not in input_encoder:
                input_encoder['path'] = hparams['data_path'] + 'encoder_' + \
                                        input_encoder['type'] + '_' + \
                                        str(size_parameter) + size_parameter_suff + '.dat'
            use_preb_embs = 'use_prebuilt_embeddings' in input_encoder and input_encoder['use_prebuilt_embeddings']
            if 'build' not in input_encoder:
                if not use_preb_embs:
                    input_encoder['build'] = os.path.isfile(input_encoder['path'])
                else:
                    input_encoder['build'] = False
            elif input_encoder['build'] and use_preb_embs:
                raise NameError('Cannot build encoder and have prebuilt embeddings')
            elif not input_encoder['build'] and use_preb_embs:
                if 'prebuilt_embeddings_path' not in input_encoder:
                    raise NameError('Need path for prebuilt embeddings')
            if input_encoder['build'] and not use_preb_embs:
                input_encoder['use_prebuilt_embeddings'] = False
            if 'bag' not in input_encoder:
                input_encoder['bag'] = False
            if 'positional_encoding' not in input_encoder:
                input_encoder['positional_encoding'] = False
            if not input_encoder['bag'] and input_encoder['positional_encoding']:
                raise NameError('Cannot have positional encoding without bags')
            if 'drop_end' not in input_encoder:
                input_encoder['drop_end'] = EncoderDefaults.drop_end
                
        collisions = 0
        for collision_target in input_encoders:
            if collision_target == input_encoder:
                collisions += 1
        if collisions > 1:
            raise NameError('Found multiple identical entries: \n%s' % input_encoder)
    if hparams['copy'] and not copy_encoder_exists:
        raise NameError('Cannot use "copy" without a "copy" input encoder')
    return hparams

