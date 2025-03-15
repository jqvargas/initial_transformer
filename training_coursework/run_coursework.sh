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

# Install dependencies if not already installed
if [ ! -f "$PYTHONUSERBASE/lib/python3.10/site-packages/protobuf" ]; then
    pip install -r requirements.txt --user
fi

# Set environment variable to use the new protobuf implementation
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb

srun -n 1 -c 10 python3 train.py --config short

