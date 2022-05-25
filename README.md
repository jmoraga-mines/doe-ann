# doe-ann
ANN for Geothermal classification

## Set up directory structure 
Create a directory structure on a base directory, that includes the Python scripts in one subdirectory and the data files in a second subdirectory.
```
[base directory]--+-- doe-ann [This repository]
                  +-- doe-data
                  +-- doe-results
```
## Run the Geothermal AI
**python geoai.py -a -d ../doe-som/brady_som_output.grd -l tmp/doe_19x3d200b.l -m tmp/doe19x3d200b.h5 -k 19 -b 200 -e 200 -c 3 -i 9 -s 100000 -r**

* Creates from scratch (-r : reset), and with data augmentation (-a), an AI with samples from the directory brady_samples_19x3d (-d ../doe-data/brady_samples_19x3d)
* It will output a labels file (-l tmp/doe_19x3d200b.l) and a model file (-m tmp/doe19x3d200b.h5). This will contain the network architecture and all weights.
* The kernel to use is 19 by 19 (-k 19), it will run batches of size 200 (-b: batch size) and will run for 200 full iterations (-e: epochs).
* The network will use 3 data bands/channels (-c 3)
* The network will have internal CNNs up to 9x9 (-i 9) (i.e., 3x3, 5x5, 7x7 and 9x9 in the inception-like module)
* The initializer will create a total of 100,000 samples to feed training and testing of the network (-s 100000)
* The network will be initialized from scratch / reset if model file exists (-r)


## Map the model results using any other input file
**python doe_ann_map.py -i ../doe-som/brady_som_output.grd -k 19 -m tmp/doe_19x3d200b.h5 -l tmp/doe_19x3d200b.l -p tmp/out.npy -o out.grd -c 3**

* Creates a map using as basis the raster from image (-i) brady_som_output.gri.
* The kernel to us is size 19x19 (-k 19)
* The model and labels are in the files doe_19x3d200b.h5 and doe_19x3d200b.l (-m tmp/doe_19x3d200b.h5 -l tmp/doe_19x3d200b.l)

## HPC batch scripts
* Under the sbatch_scripts directory, there are examples of scripts to use in an HPC environment with SLURM
