import sys

sys.path.append('../vae')
from query_paraphrases import *

def dialog_inner_batch(mu_sigmas, model_name='dialog'):
    mu_sigmas = np.array(mu_sigmas).reshape([hparams['batch_size'] * hparams['latent_size'] * 2])
    mu_sigmas_out = do_inference(server, {'mu_sigma': mu_sigmas}, model_name)
    mu_sigmas_out = np.array(mu_sigmas_out).reshape([hparams['batch_size'], hparams['latent_size'] * 2])
    return mu_sigmas_out

def dialog_end2end_batch(lines):
    mu_sigma = encode_batch(lines)
    mu_sigma[:, hparams['latent_size']:] *= 0
    mu_sigma = dialog_inner_batch(mu_sigma)
    mu_sigma[:, hparams['latent_size']:] *= 0
    answers = decode_batch(mu_sigma)
    return answers

if __name__ == '__main__':
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
        #    answers = dialog_end2end_batch(contexts)
        answers = dialog_end2end_batch([context] * hparams['batch_size'])
        for a in answers[:1]:
            print(a)
        







