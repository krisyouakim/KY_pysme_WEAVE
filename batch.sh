#!/bin/sh

# User guide: 
# Remember to set the directory where the python script is as home_dir
# Set element, stars and NLTE to match what you want synthesised
# Can be resumed just by submitting the script again

#SBATCH -t 5:00:00
#SBATCH -p node -n 10
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
python3 pysme_exec.py 10 > exec_log_10 &
#python3 pysme_exec.py 11 > exec_log_11 &
#python3 pysme_exec.py 12 > exec_log_12 &
#python3 pysme_exec.py 13 > exec_log_13 &
#python3 pysme_exec.py 14 > exec_log_14 &
#python3 pysme_exec.py 15 > exec_log_15 &
#python3 pysme_exec.py 16 > exec_log_16 &
#python3 pysme_exec.py 17 > exec_log_17 &
#python3 pysme_exec.py 18 > exec_log_18 &
#python3 pysme_exec.py 19 > exec_log_19 &
