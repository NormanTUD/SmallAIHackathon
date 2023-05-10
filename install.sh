#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

if type ml 2>&1 >/dev/null; then
	ml modenv/hiera GCC/11.2.0 OpenMPI/4.1.1 TensorFlow/2.7.1-CUDA-11.4.1 Pillow SciPy-bundle
fi

rm -rf ~/.oo_hackathon_environment

python3 -m venv ~/.oo_hackathon_environment
source ~/.oo_hackathon_environment/bin/activate

pip3 install pytz python-dateutil tqdm

if ! type ml 2>&1 >/dev/null; then
	pip3 install Pillow pandas tensorflow
fi
