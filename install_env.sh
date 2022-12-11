#!/bin/bash
# 
# Installer for package
# 
# Run: ./install_env.sh
# 

echo 'Creating Package environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate my_env
pip install -e git+https://github.com/malcolmw/pykonal@373a7d4#egg=pykonal
conda env list
echo 'Created and activated environment:' $(which python)

# check cupy works as expected
echo 'Checking cupy version and running a command...'
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

echo 'Done!'

