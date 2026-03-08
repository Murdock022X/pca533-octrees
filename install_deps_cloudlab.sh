#!/usr/bin/sh

# Install nvidia drivers prior to running
curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
sudo apt-get update -y
sudo apt-get install -y nvhpc-25-11

sudo apt install cmake libhdf5-dev hdf5-tools lmod

source /etc/profile.d/lmod.sh

module use /opt/nvidia/hpc_sdk/modulefiles/

module load nvhpc-hpcx

