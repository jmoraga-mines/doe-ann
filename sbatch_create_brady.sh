#!/bin/env bash
#SBATCH --job-name=ann_createdata_b       # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=tmoraga@gmail.com # Where to send mail      
#SBATCH --ntasks=1                    # Run on a single CPU
# Don't use #SBATCH --gres=gpu        # We need a GPU for Tensorflow
# Don't use #SBATCH -w, --nodelist=g005,c026,c028
#SBATCH --mem=32G                     # Job memory request
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
#SBATCH --output=ann_createdata_b_%j.log   # Standard output and error log
##SBATCH -p gpu       # run only in "gpu" nodes, we need aensorflow
#SBATCH -p compute    # run only in "compute" nodes, we don't need tensorflow
#SBATCH --exclusive   # run only this job
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
# srun -N1 -p gpu -t 72:00:00 --mem=32G time(python create_doe_dataset.py -i ../doe-data/brady_som_output.gri -c 5 -d brady_samples_19x3a -k 19 -s 1000)
if [ -d "brady_samples_19x3a" ]
then
   echo "Samples directory exists, deleting"
   rm -rf brady_samples_19x3a
   rm -rf brady_samples_19x3d
fi
echo "Running time(python...brady -k 19 -s 1000)"
time(python create_doe_dataset.py -i ../doe-data/brady_som_output.gri -c 3 -d brady_samples_19x3a -k 19 -s 1000)
echo "Running time(python...brady -k 19 -s 100000)"
time(python create_doe_dataset.py -i ../doe-data/brady_som_output.gri -c 3 -d brady_samples_19x3d -k 19 -s 100000)
echo "files under: brady_samples_19x3d/0"
find brady_samples_19x3d/0 -type f | wc -l
echo "files under: brady_samples_19x3d/1"
find brady_samples_19x3d/1 -type f | wc -l


## Now create the images wth 5 layers
if [ -d "brady_samples_19x5a" ]
then
   echo "Samples directory exists, deleting"
   rm -rf brady_samples_19x5a
   rm -rf brady_samples_19x5d
fi
echo "Running time(python...brady -k 19 -s 1000)"
time(python create_doe_dataset.py -i ../doe-data/brady_som_output.gri -c 3 -d brady_samples_19x5a -k 19 -s 1000)
echo "Running time(python...brady -k 19 -s 100000)"
time(python create_doe_dataset.py -i ../doe-data/brady_som_output.gri -c 3 -d brady_samples_19x5d -k 19 -s 100000)
echo "files under: brady_samples_19x5d/0"
find brady_samples_19x5d/0 -type f | wc -l
echo "files under: brady_samples_19x5d/1"
find brady_samples_19x5d/1 -type f | wc -l
