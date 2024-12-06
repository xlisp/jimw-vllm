## ----- install nvidia driver ----------
cd ~/Download/ &&  sudo ./NVIDIA-Linux-x86_64-565.57.01.run

nvidia-smi
# => | NVIDIA-SMI 565.57.01              Driver Version: 565.57.01      CUDA Version: 12.7     |

## ------ install nvcc -------
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

## Driver Installer	
sudo apt-get install -y cuda-drivers

## check 
nvcc -V
# Build cuda_12.6.r12.6/compiler.35059454_0
nvidia-smi
#| NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6     |

