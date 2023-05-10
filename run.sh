#!/bin/bash -l

#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

echo $SCRIPT_DIR

if declare -f -F ml >/dev/null; then
	ml modenv/hiera GCC/11.2.0 OpenMPI/4.1.1 TensorFlow/2.7.1-CUDA-11.4.1 Pillow SciPy-bundle
fi

if [[ ! -e "$HOME/.oo_hackathon_environment/bin/activate" ]]; then
	if [[ ! -d "$HOME/.oo_hackathon_environment/" ]]; then
		source install.sh
	else
		echo "Error: the ~/.oo_hackathon_environment already exists, but has no bin/activate. Try deleting ~/.oo_hackathon_environment and running 'bash install.sh' manually again" >&2
		exit 255
	fi
fi
source ~/.oo_hackathon_environment/bin/activate

python3 network.py $*
