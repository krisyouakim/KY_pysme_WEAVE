#!/bin/sh

# User guide: 
# Remember to set the directory where the python script is as home_dir
# Set element, stars and NLTE to match what you want synthesised
# Can be resumed just by submitting the script again

#SBATCH -t 12:00:00
#SBATCH -p node -n 1
#SBATCH -N 1

# Naming
#SBATCH -J PySME_batch_test

# Specifying project
#SBATCH -A snic2021-5-324

# Naming of output and error files
#SBATCH -o multiOut_%j.out
#SBATCH -e multiErr_%j.err

#Setting modules and loading python from pyenv with pip installed modules 
#missing on rackham
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

module purge
module load gnuparallel
pyenv shell 3.8.6

#python3 PySME_example.py test n Ba 6490_6500 > testlog
python pysme_exec.py 61
