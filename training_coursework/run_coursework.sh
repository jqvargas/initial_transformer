#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --hint=multithread
#SBATCH --gres=gpu:1

module load nvidia/cudnn/8.6.0-cuda-11.8
module load python/3.10.8-gpu
module load libsndfile/1.0.28

# Set up Python environment
export PYTHONUSERBASE=/work/mdisspt/mdisspt/$USER/python-installs
export PYTHONPATH=/work/mdisspt/mdisspt/$USER/python-installs/lib/python3.10/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/work/mdisspt/mdisspt/$USER/python-installs/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Create and set permissions for pip cache directory
mkdir -p /work/mdisspt/mdisspt/$USER/.cache/pip
chmod 700 /work/mdisspt/mdisspt/$USER/.cache/pip
export PIP_CACHE_DIR=/work/mdisspt/mdisspt/$USER/.cache/pip

# Install dependencies if not already installed
if [ ! -f "$PYTHONUSERBASE/lib/python3.10/site-packages/protobuf" ]; then
    # First try to install protobuf using pip with --no-cache-dir to avoid cache issues
    pip install --no-cache-dir protobuf==4.25.2 --user || {
        echo "Failed to install protobuf via pip, trying alternative method..."
        # If pip fails, try to use the system's protobuf
        module load protobuf/3.20.0
    }
    
    # Install other dependencies
    pip install --no-cache-dir torch==1.13.1+cu116 torchaudio==0.13.1+cu116 torchvision==0.14.1+cu116 --user
    pip install --no-cache-dir torchtext==0.14.1 torchmetrics==1.0.3 --user
    pip install --no-cache-dir pytorch-lightning==2.2.1 tensorboard==2.14.0 --user
    pip install --no-cache-dir h5py==3.10.0 numpy>=1.24.0 matplotlib>=3.7.0 ruamel.yaml>=0.17.21 --user
fi

# Set environment variable to use the new protobuf implementation
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb

# Run the training script
srun -n 1 -c 10 python3 train.py --config short

