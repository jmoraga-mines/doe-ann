#!/bin/env bash
#SBATCH --job-name=paper_b2d_map       # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmoraga@mines.edu # Where to send mail      
#SBATCH --ntasks=1                    # Run on a single CPU
# Don't use #SBATCH --gres=gpu        # We need a GPU for Tensorflow
# Don't use #SBATCH -w, --nodelist=g005,c026,c028
#SBATCH --mem=32G                     # Job memory request
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
#SBATCH --output=paper_b2d_map_%j.log   # Standard output and error log
#SBATCH -p gpu                       # run only in "gpu" nodes, we need tensorflow
#SBATCH --exclusive                  # run only this job
echo `pwd`
echo `hostname`; date
echo "Current shell:"
ps -p $$
module purge
module add libs/cuda/10.1
source ~/.bashrc
#conda init bash
#export PATH="/u/mx/fo/jmoraga/scratch/miniconda3/bin:$PATH"
cd ~/subt/doe-ann
# conda deactivate
conda activate testmini
#conda info
echo `which python`

# Checks that the directory exists
# if it does not exist, creates it
# and subdirectories if needed
if [ ! -d "paper_results" ]
then 
    mkdir -p paper_results
fi

# Checks that the model file exists
if [ -f "tmp/brady_19x3d100.h5" ]
then
   echo "Samples directory exists, running ANN mapping"
   time(python doe_ann_map.py -c 3 -k 19 -b 200 -l tmp/brady_19x3d100.l -m tmp/brady_19x3d100.h5 -g 4 -i ../doe-data/desert_som_output.gri -p paper_results/b2d_19x3d100.npy -o paper_results/b2d_19x3d100.gri)
else
   echo "Model file *brady_19x3a100.h5* does not exist, run ANN trainer"
fi

