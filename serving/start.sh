sudo killall tensorflow_model_server

# instruction https://medium.com/@brianalois/how-to-setup-tensorflow-serving-for-production-3cc2abf7efa

# tensorflow_model_server --port=9000 --model_config_file=server_config.pbtxt
# tensorflow_model_server --port=9000 --model_config_file=home/server_config.pbtxt

#sudo docker run -it -p 9000:9000 -p 9001:9001 -v ./ tensorflow/serving:latest

#sudo docker run -ti -v $(pwd):/mnt ubuntu bash

sudo docker run --runtime=nvidia -p 8500:8500 -p 8501:8501 \
    -v /home/ruslan/data/vae/light_model_data/vae/saved_model/:/home/models \
    -v /home/ruslan/code/big_progs/paraphraser/serving/:/home/configs \
    tensorflow/serving:1.9.0-gpu --port=8500 --rest_api_port=8501 --model_config_file=/home/configs/server_config.pbtxt
#    -e LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64/usr/local/cuda-9.0/lib64::/usr/local/cuda-9.0/targets/x86_64-linux/lib \

