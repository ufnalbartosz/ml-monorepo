# fix python locale error: unsupported locale setting
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales

# install pip and emacs
sudo apt-get install python-pip
sudo apt-get install emacs
pip install --user --upgrade pip

# download cudnn-8.0-linux-x64-v5.1.tgz
tar xvcf cudnn*
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/include
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

# edit .profile
export CUDA_HOME=/usr/local/cuda
export LD_LIBRATY_PATH=/usr/local/cuda/lib64

pip install --user tensorflow_gpu
pip install --user tflearn
pip install --user scipy


# clone repo
git clone https://github.com/ufnalbartosz/project_cnn


# setup script
#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
    # The 16.04 installer works with 16.10.
    curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    apt-get update
    apt-get install cuda -y
fi
