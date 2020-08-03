#!/bin/env bash
### 10000 samples per category
# 3 channels
python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 3 -d desert_samples_21x3b -s 10000 -k 21
python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 3 -d desert_samples_23x3b -s 10000 -k 23
python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 3 -d desert_samples_25x3b -s 10000 -k 25
python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 3 -d desert_samples_27x3b -s 10000 -k 27
python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 3 -d desert_samples_29x3b -s 10000 -k 29
# 5 channels
python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 5 -d desert_samples_21x5b -s 10000 -k 21
python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 5 -d desert_samples_23x5b -s 10000 -k 23
python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 5 -d desert_samples_25x5b -s 10000 -k 25
python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 5 -d desert_samples_27x5b -s 10000 -k 27
python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 5 -d desert_samples_29x5b -s 10000 -k 29
### 20000 samples per category
# 3 channels
# python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 3 -d desert_samples_21x3c -s 20000 -k 21
# python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 3 -d desert_samples_23x3c -s 20000 -k 23
# python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 3 -d desert_samples_25x3c -s 20000 -k 25
# python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 3 -d desert_samples_27x3c -s 20000 -k 27
# python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 3 -d desert_samples_29x3c -s 20000 -k 29
# 5 channels
# python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 5 -d desert_samples_21x5c -s 20000 -k 21
# python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 5 -d desert_samples_23x5c -s 20000 -k 23
# python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 5 -d desert_samples_25x5c -s 20000 -k 25
# python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 5 -d desert_samples_27x5c -s 20000 -k 27
# python create_doe_dataset.py -i ../doe-imagestacks/desert_som_output.gri -c 5 -d desert_samples_29x5c -s 20000 -k 29
