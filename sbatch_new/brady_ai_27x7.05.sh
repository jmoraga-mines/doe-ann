#!/bin/env bash
#SBATCH --job-name=paper_brady_27x7_05        # Job name
##SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=jmoraga@mines.edu # Where to send mail      
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --gres=gpu:4          # We need a GPU for Tensorflow

# Don't use #SBATCH --gres=gpu        # We need a GPU for Tensorflow
# Don't use #SBATCH -w, --nodelist=g005,c026,c028
#SBATCH --mem=32G                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=paper_brady_27x7_05_%j.log   # Standard output and error log
#SBATCH -p gpu                       # run only in "gpu" nodes, we need tensorflow
#SBATCH --exclusive                  # run only this job

# print some info 
source ~/.bashrc
echo `pwd`
echo `hostname`; date
echo "Current shell:"
ps -p $$
module purge
#module load anaconda
module load libs/cuda
#module load cudnn
#source /curc/sw/anaconda3/latest
#cd /projects/edemir@xsede.org/doe-ann
if [ "$(whoami)" == "jmoraga" ]; then
 cd /u/mx/fo/jmoraga/subt/doe-ann
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
if [ ! -d "../doe-data" ]
then 
    mkdir -p ../doe-data
fi

# Checks that the samples directory exists
if [ -f "brady_ai_stack.grd" ]
then
   echo "Samples file exists, running ANN training"
   echo "brady_som_output.grd"
   echo "python geoai_spcv.py -a -c 7 -d brady_ai_stack.grd -k 27 -b 200 -e 200 -l ../doe-data/brady_27x3d200.05.l -m ../doe-data/brady_27x7d200.05.h5 -g 4 -s 100000  -i 05 -r"
   time(python geoai_spcv.py -a -c 7 -d brady_ai_stack.grd -k 27 -b 200 -e 200 -l ../doe-data/brady_27x3d200.05.l -m ../doe-data/brady_27x7d200.05.h5 -g 4 -s 100000  -i 05 -r)
else
   echo "Samples file *brady_som_output.grd* does not exist"
fi

# Checks that the samples directory exists
# if [ -d "brady_samples_19x3d" ]
#then
   #echo "Samples directory exists, running ANN training"
 #time(python doe_geoai.py -a -c 3 -d brady_samples_19x3d -k 19 -b 200 -e 100 -l tmp/brady_19x3d100.l -m tmp/brady_19x3d100.h5 -g 4 -r)
#else
 #  echo "Samples directory *brady_samples_19x3d* does not exist, run data creator"
#fi
