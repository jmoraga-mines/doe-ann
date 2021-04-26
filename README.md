# doe-ann
ANN for Geothermal classification

## Set up directory structure 
Create a directory structure on a base directory, that includes the Python scripts in one subdirectory and the data files in a second subdirectory.
```
[base directory]--+-- doe-ann [This repository]
                  +-- doe-data
                  +-- doe-results
```
## Create DOE dataset
**python create_doe_dataset.py -i ../doe-som/brady_som_output.gri -c 3 -d ../doe-data/brady_samples_19x3d -s 100000 -k 19**

* Create a dataset using the GRI file as source for the rasters (-i ../doe-som/brady_som_output.gri).
* Creates tiles and the output is located in the brady_samples_19x3d subdirectory, under the doe-data directory.
* It creates 100,000 samples (-s 100000) and the kernel dimensions will be 19x19, with 3 channels (-c 3)


## Run the Geothermal AI
**python doe_geoai.py -a -d ../doe-data/brady_samples_19x3d -l tmp/doe_19x3d200b.l -m tmp/doe19x3d200b.h5 -k 19 -b 200 -e 200 -c 3 -r**

* Creates from scratch (-r : reset), and with data augmentation (-a), an AI with samples from the directory brady_samples_19x3d (-d ../doe-data/brady_samples_19x3d)
* It will output a labels file (-l tmp/doe_19x3d200b.l) and a model file (-m tmp/doe19x3d200b.h5). This will contain the network architecture and all weights.
* The kernel to use is 19 by 19 (-k 19), it will run batches of size 200 (-b: batch size) and will run for 200 full iterations (-e: epochs).
* The Network will use 3 data bands/channels (-c 3)


## Map the model results using any other input file
**python doe_ann_map.py -i ../doe-som/brady_som_output.gri -k 19 -m tmp/doe_19x3d200b.h5 -l tmp/doe_19x3d200b.l**

* Creates a map using as basis the raster from image (-i) brady_som_output.gri.
* The kernel to us is size 19x19 (-k 19)
* The model and labels are in the files doe_19x3d200b.h5 and doe_19x3d200b.l (-m tmp/doe_19x3d200b.h5 -l tmp/doe_19x3d200b.l)

## HPC batch scripts
* Under the sbatch_scripts directory, there are examples of scripts to use in an HPC environment with SLURM
