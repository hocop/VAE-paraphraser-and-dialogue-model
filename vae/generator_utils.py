import numpy as np
import tensorflow as tf
import time
from multiencoder_utils import Multiencoder, NormalizeEncoderSettings

# Generator that yields batches of variable size from files with source and answers
# size of batch is defined by hparams['tokens_per_batch']
# Use this for training
def io_input_fn(so_file, a_file, hparams):
    types_dict = {'so': tf.int32}
    shapes_dict = {'so': [None, None]}
    dataset = tf.data.Dataset.from_generator(
            generator=lambda: _io_generator(so_file, a_file, hparams),
            output_types=(types_dict, tf.int32),
            output_shapes=(shapes_dict, [None, None]))
    dataset = dataset.batch(1) # 1 because generator makes batches, not this function
    return dataset

def _io_generator(so_file, a_file, hparams):
    # Load encoders
    input_encoder = Multiencoder(hparams['input_encoders'], 
            hparams['max_source_len'], 
            'tokenizer_emit_spaces' in hparams and hparams['tokenizer_emit_spaces'],
            'tokenizer_omit_ending_space' in hparams and hparams['tokenizer_omit_ending_space'])
    output_encoder = Multiencoder([hparams['output_encoder']], hparams['max_answer_len'])
    
    def generate_batch(pairs):
        # generate batches
        index0 = min([index for index in pairs])
        lens0 = pairs[index0]['lens']
        dists_dict = {np.sum(np.abs(pairs[index]['lens'] - lens0)) + np.random.random() / 100:
                index for index in pairs if index != index0} # random to make them uniq (meh)
        dists = sorted(list(set([d for d in dists_dict]))) # lowest to highest
        mli, mlo = pairs[index0]['lens']
        sum_words = mli + mlo
        batch_so = [pairs[index0]['so_encoded']]
        batch_a = [pairs[index0]['a_encoded']]
        del pairs[index0]
        for d in dists:
            so = pairs[dists_dict[d]]['so_encoded']
            a = pairs[dists_dict[d]]['a_encoded']
            max_len_in = np.max([mli, pairs[dists_dict[d]]['lens'][0]])
            max_len_out = np.max([mlo, pairs[dists_dict[d]]['lens'][1]])
            words_count = (max_len_in + max_len_out) * (len(batch_so) + 1)
            if words_count > hparams['tokens_per_batch'] and len(batch_so) != 0:
                break
            sum_words += pairs[dists_dict[d]]['lens'][0] + pairs[dists_dict[d]]['lens'][1]
            mli = max_len_in
            mlo = max_len_out
            batch_so.append(so)
            batch_a.append(a)
            del pairs[dists_dict[d]]
        percent_words = sum_words / ((mli + mlo) * len(batch_so)) # 0 to 1. more - better
        return {'so': np.array(batch_so)}, np.array(batch_a)
    
    # get lines
    pairs = {}
    for this_index, (so_l, a_l) in enumerate(zip(so_file, a_file)):
        so, a = so_l.strip(), a_l.strip()
        if len(so) == 0 or len(a) == 0:
            continue
        if len(pairs) < hparams['max_chunk_size']:
            try:
                so_encoded = input_encoder.encode(so)
                a_encoded = output_encoder.encode(a, emit_eos=True)
            except:
                print('Warning: failed to encode pair (check hparams max_source_len and max_answer_len)')
                continue
            pairs[this_index] = {
                'so': so,
                'a': a,
                'so_encoded': so_encoded,
                'a_encoded': a_encoded,
                'lens': np.array([np.sum(np.sign(so_encoded[:hparams['max_source_len']])),
                                    np.sum(np.sign(a_encoded[:hparams['max_answer_len']]))])
            }
            continue
        if this_index % 1e5 == 0:
            pairs = {index: pairs[index] for index in pairs}
        yield generate_batch(pairs)
    yield generate_batch(pairs)


# Generator that yields batches of hparams['batch_size'] from input file
# Use this for inference
def i_input_fn(so_file, hparams):
    types_dict = {'so': tf.int32}
    shapes_dict = {'so': [None, None]}
    dataset = tf.data.Dataset.from_generator(
            generator=lambda: _i_generator(so_file, hparams),
            output_types=types_dict,
            output_shapes=shapes_dict)
    dataset = dataset.batch(1) # 1 because generator makes batches, not this function
    return dataset

def _i_generator(so_file, hparams):
    # get encoder
    input_encoder = Multiencoder(hparams['input_encoders'], 
            hparams['max_source_len'], 
            'tokenizer_emit_spaces' in hparams and hparams['tokenizer_emit_spaces'],
            'tokenizer_omit_ending_space' in hparams and hparams['tokenizer_omit_ending_space'])
    # generate batches
    batch_so = []
    for so_l in so_file:
        so_l = so_l.strip()
        batch_so.append(input_encoder.encode(so_l))
        if len(batch_so) == hparams['batch_size']:
            yield {'so': np.array(batch_so)}
            batch_so = []
    if len(batch_so) != 0:
        yield {'so': np.array(batch_so)}
