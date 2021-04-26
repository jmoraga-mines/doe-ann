#!/bin/env bash
#SBATCH --job-name=ann_createdata_d   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmoraga@mines.edu # Where to send mail      
#SBATCH --ntasks=1                    # Run on a single CPU
# Don't use #SBATCH --gres=gpu        # We need a GPU for Tensorflow
# Don't use #SBATCH -w, --nodelist=g005,c026,c028
#SBATCH --mem=32G                     # Job memory request
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
#SBATCH --output=ann_createdata_d_%j.log   # Standard output and error log
##SBATCH -p gpu                       # run only in "gpu" nodes, we need tensorflow
#SBATCH -p compute                       # run only in "compute" nodes, we don't need tensorflow
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
# srun -N1 -p gpu -t 72:00:00 --mem=32G time(python create_doe_dataset.py -i ../doe-data/desert_som_output.gri -c 5 -d desert_samples_19x3a -k 19 -s 1000)
if [ -d "desert_samples_19x3a" ]
then
   echo "Samples directory exists, deleting"
   rm -rf desert_samples_19x3a
   rm -rf desert_samples_19x3d
fi
echo "Running time(python...desert -k 19 -s 1000)"
time(python create_doe_dataset.py -i ../doe-data/desert_som_output.gri -c 3 -d desert_samples_19x3a -k 19 -s 1000)
echo "Running time(python...desert -k 19 -s 100000)"
time(python create_doe_dataset.py -i ../doe-data/desert_som_output.gri -c 3 -d desert_samples_19x3d -k 19 -s 100000)
echo "files under: desert_samples_19x3d/0"
find desert_samples_19x3d/0 -type f | wc -l
echo "files under: desert_samples_19x3d/1"
find desert_samples_19x3d/1 -type f | wc -l


## Now create the images wth 5 layers
if [ -d "desert_samples_19x5a" ]
then
   echo "Samples directory exists, deleting"
   rm -rf desert_samples_19x5a
   rm -rf desert_samples_19x5d
fi
echo "Running time(python...desert -k 19 -s 1000)"
time(python create_doe_dataset.py -i ../doe-data/desert_som_output.gri -c 3 -d desert_samples_19x5a -k 19 -s 1000)
echo "Running time(python...desert -k 19 -s 100000)"
time(python create_doe_dataset.py -i ../doe-data/desert_som_output.gri -c 3 -d desert_samples_19x5d -k 19 -s 100000)
echo "files under: desert_samples_19x5d/0"
find desert_samples_19x5d/0 -type f | wc -l
echo "files under: desert_samples_19x5d/1"
find desert_samples_19x5d/1 -type f | wc -l
