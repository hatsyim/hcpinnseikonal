#!/bin/bash
# 
# Installer for package
# 
# Run: ./install_env.sh

echo 'Creating Package environment'

# Create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hcpinnseikonal

# Install hcpinnseikonal package
pip install -e .

# Install packages for 3D plotting on Jupyterlab
pip install pyvista[all]
pip install pykrige
pip install pykonal

conda env list
echo 'Created and activated environment:' $(which python)

# Check cupy works as expected
echo 'Checking cupy version and running a command...'
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

echo 'Done!'
