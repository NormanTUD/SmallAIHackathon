#!/bin/bash -l

#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

echo $SCRIPT_DIR

if type ml 2>&1 >/dev/null; then
	ml modenv/hiera GCC/11.2.0 OpenMPI/4.1.1 TensorFlow/2.7.1-CUDA-11.4.1 Pillow SciPy-bundle
fi

if [[ ! -e "$HOME/.oo_hackathon_environment/bin/activate" ]]; then
	source install.sh
fi
source ~/.oo_hackathon_environment/bin/activate

python3 network.py $*
