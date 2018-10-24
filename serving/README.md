#How to Setup Tensorflow Serving on GPU

1) Make sure usual tensorflow-gpu is working  
    `pip3 show tensorflow-gpu`
2) Install docker-ce and nvidia-docker. If nvidia-docker gives error, it will show proper version of docker-ce. Install it.  
3) Go to directory where you want to store tf-server source and run  
    `git clone https://github.com/tensorflow/serving`  
    `cd serving/tensorflow_serving/tools/docker`  
    `docker build --pull -t tensorflow/serving:1.9.0-gpu -f Dockerfile.gpu . # insert your version of tensorflow-gpu`  
4) To run docker set your paths in `start.sh` and launch it  
