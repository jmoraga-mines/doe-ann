#!/bin/env bash
#SBATCH --job-name=ann_createdata_d       # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=edemir@mines.edu # Where to send mail      
#SBATCH --ntasks=1                    # Run on a single CPU
# Don't use #SBATCH --gres=gpu        # We need a GPU for Tensorflow
# Don't use #SBATCH -w, --nodelist=g005,c026,c028
#SBATCH --mem=32G                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=ann_createdata_d_%j.log   # Standard output and error log
#SBATCH -p sgpu       # run only in "gpu" nodes, we need aensorflow
# Dont use #SBATCH -p compute    # run only in "compute" nodes, we don't need tensorflow
#SBATCH --exclusive   # run only this job

# print some info 
echo `pwd`
echo `hostname`; date
echo "Current shell:"
ps -p $$
module purge
module load anaconda
module load cuda/10.2
module load cudnn
source /curc/sw/anaconda3/latest
cd /projects/edemir@xsede.org/doe-ann
conda activate doe-env
# print some more info 
echo `which python`
echo `pwd`
echo `nvidia-smi`

if [ -d "../samples/desert" ]
then
   echo "Samples directory exists, deleting"
   rm -rf ../samples/desert/s19x3x1000
   rm -rf ../samples/desert/s19x3x100000
   rm -rf ../samples/desert/s19x5x1000
   rm -rf ../samples/desert/s19x5x100000
fi

echo "Running time(python...desert -k 19 -s 1000 -c 3)"
time(python create_doe_dataset.py -i ../doe-som/desert_som_output.gri -c 3 -d ../samples/desert/s19x3x1000 -k 19 -s 1000)
echo "files under: samples/desert/s19x3x1000/0"
find ../samples/desert/s19x3x1000/0 -type f | wc -l
echo "files under: samples/desert/s19x3x1000/1"
find ../samples/desert/s19x3x1000/1 -type f | wc -l

echo "Running time(python...desert -k 19 -s 1000 -c 5)"
time(python create_doe_dataset.py -i ../doe-som/desert_som_output.gri -c 5 -d ../samples/desert/s19x5x1000 -k 19 -s 1000)
echo "files under: samples/desert/s19x5x1000/0"
find ../samples/desert/s19x5x1000/0 -type f | wc -l
echo "files under: samples/desert/s19x5x1000/1"
find ../samples/desert/s19x5x1000/1 -type f | wc -l

echo "Running time(python...desert -k 19 -s 100000 -c 3)"
time(python create_doe_dataset.py -i ../doe-som/desert_som_output.gri -c 3 -d ../samples/desert/s19x3x100000 -k 19 -s 100000)
echo "files under: samples/desert/s19x3x100000/0"
find ../samples/desert/s19x3x100000/0 -type f | wc -l
echo "files under: samples/desert/s19x3x100000/1"
find ../samples/desert/s19x3x100000/1 -type f | wc -l

echo "Running time(python...desert -k 19 -s 100000 -c 5)"
time(python create_doe_dataset.py -i ../doe-som/desert_som_output.gri -c 5 -d ../samples/desert/s19x5x100000 -k 19 -s 100000)
echo "files under: samples/desert/s19x5x100000/0"
find ../samples/desert/s19x5x100000/0 -type f | wc -l
echo "files under: samples/desert/s19x5x100000/1"
find ../samples/desert/s19x5x100000/1 -type f | wc -l

echo "finished :)"
