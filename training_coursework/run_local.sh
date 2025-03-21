#!/bin/bash

# Set up Python environment
export PYTHONUSERBASE=/work/m24ol/m24ol/$USER/python-installs
export PYTHONPATH=/work/m24ol/m24ol/$USER/python-installs/lib/python3.10/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/work/m24ol/m24ol/$USER/python-installs/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Add PyTorch paths
export PYTHONPATH=$PYTHONPATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-gpu/python/3.10.8/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-gpu/python/3.10.8/lib
export LIBRARY_PATH=$LIBRARY_PATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-gpu/python/3.10.8/lib

# Set protobuf environment variables
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2

# Install dependencies if needed
pip install --user -r ../requirements.txt

# Ensure our local protobuf is used
export PYTHONPATH=/work/m24ol/m24ol/$USER/python-installs/lib/python3.10/site-packages:$PYTHONPATH

# Run the training script
python3 train.py --config short 