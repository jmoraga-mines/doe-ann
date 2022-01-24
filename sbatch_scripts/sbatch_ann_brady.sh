#!/bin/env bash
#SBATCH --job-name=paper_brady_100       # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=edemir@mines.edu # Where to send mail      
#SBATCH --ntasks=1                    # Run on a single CPU
# Don't use #SBATCH --gres=gpu        # We need a GPU for Tensorflow
# Don't use #SBATCH -w, --nodelist=g005,c026,c028
#SBATCH --mem=32G                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=paper_brady_100_%j.log   # Standard output and error log
#SBATCH -p sgpu                       # run only in "gpu" nodes, we need tensorflow
#SBATCH --exclusive                  # run only this job
echo `pwd`
echo `hostname`; date
echo "Current shell:"
ps -p $$
module purge
module load anaconda
# module add libs/cuda/10.1  # This does not work in XSEDE, only in Wendian
module load cuda/10.2
module load cudnn
#source ~/.bashrc
source /curc/sw/anaconda3/latest
#conda init bash
#export PATH="/u/mx/fo/jmoraga/scratch/miniconda3/bin:$PATH"
cd /projects/edemir@xsede.org/doe-ann
#cd ~/subt/doe-ann
# conda deactivate
conda activate doe-env
#conda info
echo `which python`
echo `pwd`
echo `nvidia-smi`
# Checks that the directory exists
# if it does not exist, creates it
# and subdirectories if needed
if [ ! -d "tmp" ]
then 
    mkdir -p tmp
fi

# Checks that the samples directory exists
if [ -d "../samples/brady/s19x3x1000" ]
then
   echo "Samples directory exists, running ANN training"
   time(python doe_geoai.py --gpus=4 -a -c 3 -d ../samples/brady/s19x3x1000 -k 19 -b 200 -e 100 -l ../doe-data/brady_s19x3x1000.l -m ../doe-data/brady_s19x3x1000.h5 -g 4 -r)
else
   echo "Samples directory *s19x3x1000* does not exist, run data creator"
fi

# Checks that the samples directory exists
#3if [ -d "brady_samples_19x3d" ]
#then
   #echo "Samples directory exists, running ANN training"
 #time(python doe_geoai.py -a -c 3 -d brady_samples_19x3d -k 19 -b 200 -e 100 -l tmp/brady_19x3d100.l -m tmp/brady_19x3d100.h5 -g 4 -r)
#else
 #  echo "Samples directory *brady_samples_19x3d* does not exist, run data creator"
#fi
