### Building:  
####    1) set hyperparameters in json file
Copy default hparams file from ./hparams/hparams_default.json  
Edit copied file (especially "path_..." fields)  
See description of all fields below  
####    2) run prepare_bank.py hparams.json
This script converts dialogs from "text_path" to context-reply pairs in "pairs_path"  
You may use your own version of this script for your specific problem  
####    3) run build_encoders.py hparams.json
This script builds the needed input and output encoder vocabularies  
####    4) run train.py hparams.json
Do training iterations (set "num_epochs") and save checkpoints to hparams["checkpoints_path"]  
####    5) run predict.py hparams.json
Do predictions on train set and write them to hparams["answers_path"]/answers.txt  
Saved checkpoints are used  
Optionally p_gen.txt and confidence.txt may be created (if set in hparams)  
####    6) python3 print.py hparams.json
Make visual html representation of answers. Copied words are highlighted blue  
Thist script also prints accuracy and BLEU of predictions in "answers_path"  

### Handle results
####    Make diff between several models
    python3 print.py hparams_model_1.json hparams_model_2.json hparams_model_N.json  
####    Make html overview of all your models
    1) set paths where you store all hparams in framework_config.json  
    2) run  
    python3 overview.py hparams/framework_config.json  

### Serving:
"export.py" saves model graph and weights to ./saved_model/  
#### To run server:  
    tensorflow_model_server --port=9000 --model_name=s2s_hist_copy --model_base_path='/home/ruslan/Code/nb_research_sbs/seq2seq_bahd_hist_copy_subwords/saved_model'  
#### To test quering:  
    python3 query_bank.py  

### Freezing model to .pb to use in java server:
    first, do a trick:  
    set "freezing_mode" in json to true  
    run train.py and abort after saving first checkpoint  
    run freeze.py  
    don't forget to set "freezing_mode" back to false  

## How to use hparams:
hyperparameters are stored as json vocabulary.  
Macros:  
1) if value is string, value of other field can be inserted:  
    "model_name": "seq2seq",  
    "hidden_size": 128,  
    "answers_path": "/home/<user_name>/data/light_model_data/<model_name>/answers/"  
2) "user_name" field is created automaticaly if not specified
3) json can be derived from another json. Can be useful for series of experiments with small changes.  
This example overrides model_name and attention_scheme:  
    {  
        "model_name": "seq2seq_with_bahdanau",  
        "derive_from": "hparams/hparams_default.json",  
        "attention_scheme": "bahdanau"  
    }  

## Hyperparameters description:
0) Global parameters  
"model_name": descriptive name of model. Please don't use '+' character  
"user_name": unix user name. Set "default" to automatically define it  
"experiment_series": name of experiment. Helpful if using overview.py  
1) Architecture:  
"freezing_mode": if true, builds graph with inference and training parts at the same time and disables dropout. Recommended to set false.  
"num_layers": number of GRU layers  
"attention_scheme": type of attention mechanism. Can be "None"; "bahdanau"(https://arxiv.org/pdf/1409.0473.pdf, for multi-layer network, all layers outputs are used); "luong"(https://arxiv.org/pdf/1508.04025.pdf); "google-nmt"(https://arxiv.org/pdf/1609.08144.pdf, like bahdanau but uses only bottom layer output => lighter); "mixed" - bahdanau and luong in one: sends context vector both to projection layer and to next decoder cell (see mixed_attention_mechanism.jpg red arrows)  
"attention_score": scoring function. Can be "dot", "general", "concat". (See Luong article https://arxiv.org/pdf/1508.04025.pdf)  
"bidirectional_encoder": first layer of encoder becomes bidirectional  
"copy": to use copy mechanism or not. Works only if attention is not "None" (article https://arxiv.org/pdf/1704.04368.pdf)  
"history": to use history mechanism or not. Works only if attention is not "None"  
"encoder_decoder_share_weights": share weights between encoder and decoder rnn cells  
2) Model parameters:  
"hidden_size": size of hidden layer  
"embedding_size": word embedding size  
"attention_vector_size": size of attention comparing vector. "default" - use hidden_size  
"vocab_size": vocabulary size  
"dropout_rate": dropout during training  
"history_vector_size": parameter of history mechanism  
"history_convkernel_size": parameter of history mechanism  
3) Loss calculation parameters:  
"loss_normalize_by_length": divide loss by answer length. Note: when this parameter is false, you may have to lower learning_rate  
4) Prediction parameters:  
"use_beam_search": use beam search if true, else greedy  
"beam_width": beam width if beam search is used  
"length_penalty_weight": parameter of beamsearch  
"batch_size": batch size used on evaluation and inference, not training  
"draw_attention": visualization of attention mechanism during inference  
"draw_entropy": plot entropy of attention distribution after inference  
5) Training parameters  
"log_bleu_every": how often to predict and calculate bleu (if not zeros)
"tokens_per_batch": batch size for training, defined by number of tokens, not examples (in tensor2tensor this is called just batch_size). For training on the whole dataset, this parameter must not be smaller then "max_source_length"  
"num_epochs": number of epochs  
"learning_rate": learning rate  
"keep_checkpoint_max": maximum number of checkpoints to keep saved on disk  
"label_smoothing": small number. What is it: https://arxiv.org/pdf/1706.03762.pdf; https://arxiv.org/pdf/1512.00567.pdf  
6) Filesystem  
"text_path": path to dialogs files  
"pairs_path": path to paired text data  
"data_path": path to numpy data  
"checkpoints_path": path to store checkpoints data  
"answers_path": where to store all predictions  
7) Data preparation parameters  
"max_source_len": maximum number of tokens in input source  
"max_answer_len": maximum length of answer  
"input_sorting_noise_level": how much to shuffle the data. 0 - fastest training. Use 1000000 if not sure that sorting data will not affect results  
"output_sorting_noise_level": how much to shuffle the data. Use 0 when answers are very sparse (chit-chat, translation). Use around 10 if there are a lot of identical answers (most QA systems). The more noise level you set - the slower will be training process  
8) "encoder_settings": dict contatining all the input and output encoder settings
* "input_encoders": list of encodings to use, can be any of "char", "2char", "3char", "wordpiece", "token". Emission of encodings will follow the order from shortest to longest
* "token_vocab": path to the intermediate vocabulary for the input encoder
* "input_encoder_path": path to the final input encoder vocab
* "output_encoder": encoding to use for answers. Can be any of "char", "wordpiece"
* "output_encoder_path": path of the output encoder vocab
* "wordpiece_target_vocab_size": target size for any wordpiece vocabularies (input and output)
* "X_encoder_cut": freqency, below or equal to which all X from train set will not make it to token_vocab if "X" is in "input_encoders"
* "input_delimiters": list of delimiters used in pairs. Usually "|", "SYSTEM" or "CLIENT". Used to build a more balanced token vocab.
