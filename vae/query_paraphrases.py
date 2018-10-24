from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import classification_pb2
from multiencoder_utils import Multiencoder, NormalizeEncoderSettings
from load_hparams import hparams
hparams = NormalizeEncoderSettings(hparams)

# if run over ssh, run `stty iutf8` to fix backspace deleting always 1 byte

server = '127.0.0.1:8500'
max_source_length = 1000

def make_seq_example(input_ids, feature_name="so"):
    features = {
            feature_name:
                tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids))
    }
    return tf.train.Example(features=tf.train.Features(feature=features))

def make_dense_example(input_ids, feature_name="mu_sigma"):
    features = {
            feature_name:
                tf.train.Feature(float_list=tf.train.FloatList(value=input_ids))
    }
    return tf.train.Example(features=tf.train.Features(feature=features))

def do_inference(hostport, features, model_name):
    if 'so' in features:
        data_so = make_seq_example(features['so'])
    else:
        data_so = make_dense_example(features['mu_sigma'])
    host, port = hostport.split(':')
    
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'output'
    request.inputs['examples'].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                    data_so.SerializeToString(),
                    shape=[1]))
    result = stub.Predict(request, 100.0)  # 10 secs timeout
    if result.outputs['output'].int64_val != []:
        return result.outputs['output'].int64_val
    elif result.outputs['output'].int_val != []:
        return result.outputs['output'].int_val
    else:
        return result.outputs['output'].float_val

encoder_in = Multiencoder(hparams['input_encoders'], hparams['max_source_len'])
encoder_out = Multiencoder([hparams['output_encoder']], hparams['max_answer_len'])

def encode_batch(lines, model_name='vae_encoder'):
    sources = []
    if len(lines) != hparams['batch_size']:
        raise BaseException('number of lines must be equal batch size')
    for l in lines:
        source = encoder_in.encode(l)
        source = np.concatenate((source, np.zeros(max_source_length - len(source), 'int32')))
        sources.append(source)
    sources = np.concatenate(sources)
    mu_sigmas = do_inference(server, {'so': sources}, model_name)
    mu_sigmas = np.array(mu_sigmas).reshape([hparams['batch_size'], hparams['latent_size'] * 2])
    return mu_sigmas

def decode_batch(mu_sigmas, model_name='vae_decoder'):
    mu_sigmas = np.array(mu_sigmas).reshape([hparams['batch_size'] * hparams['latent_size'] * 2])
    answers = do_inference(server, {'mu_sigma': mu_sigmas}, model_name)
    answers = np.array(answers).reshape([hparams['batch_size'], len(answers) // hparams['batch_size']])
    a = []
    for answer in answers:
        answer = list(answer)
        if 1 in answer: # 1 - index of <EOS>
            answer = answer[:answer.index(1)]
        answer = encoder_out.decode(answer)
        a.append(answer)
    return a

if __name__ == '__main__':
    # make one test request
    source = [1] + [0 for i in range(max_source_length - 1)]
    source = np.concatenate([source] * hparams['batch_size'])
    do_inference(server, {'so': source}, 'vae_encoder')
    print('system is ready')
    contexts = []
    while True:
        inputs = input(">> ")
        if inputs.lower() == "end":
            context = []
            print("\nдиалог завершен\n")
            continue
        if inputs != "":
            context = inputs.lower().strip()
        contexts.append(context)
        #if len(contexts) == 4:
        #    mu_sigmas = encode_batch(contexts)
        mu_sigmas = encode_batch([context] * hparams['batch_size'])
        answers = decode_batch(mu_sigmas)
        for a in answers:
            print(a)
        







