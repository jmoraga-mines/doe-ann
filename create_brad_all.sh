#!/bin/env bash
### 10,000 samples per category
# 3 channels
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_21x3b -s 10000 -k 21
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_23x3b -s 10000 -k 23
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_25x3b -s 10000 -k 25
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_27x3b -s 10000 -k 27
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_29x3b -s 10000 -k 29
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_31x3b -s 10000 -k 31
# 5 channels
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_21x5b -s 10000 -k 21
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_23x5b -s 10000 -k 23
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_25x5b -s 10000 -k 25
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_27x5b -s 10000 -k 27
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_29x5b -s 10000 -k 29
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_31x5b -s 10000 -k 31
### 20,000 samples per category
# 3 channels
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_21x3c -s 20000 -k 21
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_23x3c -s 20000 -k 23
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_25x3c -s 20000 -k 25
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_27x3c -s 20000 -k 27
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_29x3c -s 20000 -k 29
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_31x3c -s 20000 -k 31
# 5 channels
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_21x5c -s 20000 -k 21
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_23x5c -s 20000 -k 23
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_25x5c -s 20000 -k 25
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_27x5c -s 20000 -k 27
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_29x5c -s 20000 -k 29
# python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_31x5c -s 20000 -k 31
### 100,000 samples per category
# 3 channels
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_21x3d -s 100000 -k 21
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_23x3d -s 100000 -k 23
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_25x3d -s 100000 -k 25
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_27x3d -s 100000 -k 27
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_29x3d -s 100000 -k 29
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 3 -d brady_samples_31x3d -s 100000 -k 31
# 5 channels
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_21x5d -s 100000 -k 21
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_23x5d -s 100000 -k 23
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_25x5d -s 100000 -k 25
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_27x5d -s 100000 -k 27
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_29x5d -s 100000 -k 29
python create_doe_dataset.py -i ../doe-imagestacks/brady_som_output.gri -c 5 -d brady_samples_31x5d -s 100000 -k 31
