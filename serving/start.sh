sudo killall tensorflow_model_server

sudo docker run --runtime=nvidia -p 8500:8500 -p 8501:8501 \
    -v /home/ruslan/data/vae/light_model_data/vae/saved_model/:/home/models \
    -v /home/ruslan/code/big_progs/paraphraser/serving/:/home/configs \
    tensorflow/serving:1.9.0-gpu --port=8500 --rest_api_port=8501 --model_config_file=/home/configs/server_config.pbtxt
