#!/bin/env bash
# @author: jmoraga@mines.edu
# (c) 2021, 2022 Jaime Moraga
#SBATCH --job-name=paper_brady_27x7_13        # Job name
##SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=jmoraga@mines.edu # Where to send mail      
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --gres=gpu:4          # We need a GPU for Tensorflow

# Don't use #SBATCH --gres=gpu        # We need a GPU for Tensorflow
# Don't use #SBATCH -w, --nodelist=g005,c026,c028
#SBATCH --mem=64G                     # Job memory request
##SBATCH --mem-per-cpu=64G             # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=paper_brady_27x7_13_%j.log   # Standard output and error log
#SBATCH -p gpu                       # run only in "gpu" nodes, we need tensorflow
#SBATCH --exclusive                  # run only this job

# print some info 
source ~/.bashrc
echo `pwd`
echo `hostname`; date
echo "Current shell:"
ps -p $$
module purge
if [ "$(whoami)" == "jmoraga" ]; then
 module load libs/cuda
 cd /u/mx/fo/jmoraga/subt/doe-ann
elif [ "$(whoami)" == "edemir" ]; then
 module load anaconda
 module load cudnn
 source /curc/sw/anaconda3/latest
 cd /projects/edemir@xsede.org/doe-ann
fi
conda init bash
conda activate tf_38
# print some more info 
echo `which python`
echo `pwd`
echo `nvidia-smi`

# Checks that the directory exists
# if it does not exist, creates it
# and subdirectories if needed
if [ ! -d "cv_results" ]
then 
    mkdir -p cv_results
fi

   echo "python GridSearch.py "

   time(python GridSearch.py)
