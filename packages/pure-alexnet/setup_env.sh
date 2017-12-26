# fix python locale error: unsupported locale setting
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales

# download and install conda
curl -O https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
bash Anaconda3-4.2.0-Linux-x86_64.sh

# prepare conda env
conda create -n tensorflow python=3.6

# prepare tensorflow with GPU support
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install build-essential

sudo apt-get install python-dev python3-dev python-virtualenv

# install nvidia driver
sudo apt-get install nvidia-387
sudo apt-get install mesa-common-dev
sudo apt-get install freeglut3-dev

# install cuda 8.0
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
sudo bash cuda_8.0.61_375.26_linux-run

# export cuda paths
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# install cudnn
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/cudnn-8.0-linux-x64-v6.0-tgz
tar -zxvf cudnn*
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

# install tf-gpu
source activate tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp36-cp36m-linux_x86_64.whl
