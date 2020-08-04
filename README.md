# doe-ann
ANN for Geothermal classification

## create_doe_dataset.py
**python create_doe_dataset.py -i ../doe-som/brady_som_output.gri -c 3 -d brady_samples_19x3d -s 100000 -k 19**

* Create a dataset using the GRI file as source for the rasters. Creates tiles with 3 channels (bands),and the output is located in the directory brady_samples_19x3d. 
* It creates 100,000 samples and the kernel dimensions will be 19x19


## doe_ann3.py
**python doe_ann3.py -a -d brady_samples_19x3d -l tmp/doe_19x3d200b.l -m tmp/doe19x3d200b.h5 -k 19 -b 200 -e 200 -c 3 -r **

* Creates from scratch (-r : reset), and with data augmentation (-a), an AI with samples from the directory brady_samples_19x3d. It will output a lables file doe_19x3d200b.l and a model file doe19x3d200b.h5 (this will contain the network architecture and all weights).
* The kernel to use is 19x19, it will run batches of size 200 (-b: batch size) and will runfor 200 full iterations (-e: epochs). The Netwrok will use 3 data bands."


## doe_ann_map.py
**python doe_ann_map.py -i ../doe-som/brady_som_output.gri -k 19 -m tmp/doe_19x3d200b.h5 -l tmp/doe_19x3d200b.l **

* Creates a map using as basis the raster from image (-i) brady_som_output.gri. The kernel to us is size 19x19, and the model and labels are in the files doe_19x3d200b.h5 and doe_19x3d200b.l
