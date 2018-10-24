import numpy as np
import tensorflow as tf
import sys

sys.path.append('../vae')
from dialog_model import get_model_fn

# run server command:
# tensorflow_model_server --port=9000 --model_name=s2s_hist_copy --model_base_path='/home/ruslan/Code/nb_research_master/models/seq2seq_hist_copy/saved_model/'

from load_hparams import hparams
model_fn = get_model_fn(hparams)
export_path = hparams['export_path']

def main(unused_argv):
    # Create the Estimator
    estimator = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=hparams["checkpoints_path"])
    
    # Export dialog model
    feature_spec = {
        "mu_sigma": tf.FixedLenFeature(dtype=tf.float32,
                    shape=[hparams['batch_size'], 2 * hparams['latent_size']]),
    }
    def serving_input_receiver_fn():
        """An input receiver that expects a serialized tf.Example."""
        serialized_tf_example = tf.placeholder(dtype=tf.string,
                shape=[1], # batch_size
                name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    
    estimator.export_savedmodel(export_path + 'dialog', serving_input_receiver_fn,
                            strip_default_attrs=True)

if __name__ == "__main__":
    tf.app.run()




















