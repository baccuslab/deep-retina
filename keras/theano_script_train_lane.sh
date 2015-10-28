# Run this on the Rye clusters, after first running the command "module load cuda cudasamples"
CUDA_ROOT=$CUDA_ROOT:/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
THEANO_FLAGS=device=gpu0,floatX=float32 python train-3_layer-convnet.py
