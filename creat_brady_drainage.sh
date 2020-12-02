#!/bin/env bash
### 10,000 samples per category
# 6 channels
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_21x3b -s 10000 -k 21
#python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_23x6b -s 10000 -k 23
#python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_25x6b -s 10000 -k 25
python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_27x6b -s 10000 -k 27
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_29x3b -s 10000 -k 29
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_31x3b -s 10000 -k 31

# 5 channels
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_21x5b -s 10000 -k 21
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_23x5b -s 10000 -k 23
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_25x5b -s 10000 -k 25
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_27x5b -s 10000 -k 27
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_29x5b -s 10000 -k 29
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_31x5b -s 10000 -k 31
### 20,000 samples per category
# 3 channels
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 3 -d brady_basic_21x3c -s 20000 -k 21
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 3 -d brady_basic_23x3c -s 20000 -k 23
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 3 -d brady_basic_25x3c -s 20000 -k 25
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 3 -d brady_basic_27x3c -s 20000 -k 27
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 3 -d brady_basic_29x3c -s 20000 -k 29
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 3 -d brady_basic_31x3c -s 20000 -k 31
# 5 channels
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_21x5c -s 20000 -k 21
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_23x5c -s 20000 -k 23
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_25x5c -s 20000 -k 25
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_27x5c -s 20000 -k 27
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_29x5c -s 20000 -k 29
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 5 -d brady_slope_31x5c -s 20000 -k 31

### 100,000 samples per category
# 6 channels
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_21x3d -s 100000 -k 21
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_23x3d -s 100000 -k 23
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_25x3d -s 100000 -k 25
python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_27x6d -s 100000 -k 27
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_29x3d -s 100000 -k 29
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 6 -d brady_drain_31x3d -s 100000 -k 31
# 7 channels
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 7 -d brady_aspec_21x5d -s 100000 -k 21
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 7 -d brady_aspec_23x5d -s 100000 -k 23
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 7 -d brady_aspec_25x5d -s 100000 -k 25
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 7 -d brady_aspec_27x7d -s 100000 -k 27
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 7 -d brady_aspec_29x5d -s 100000 -k 29
# python create_doe_dataset.py -i ../rasters/brady_trainer.gri -c 7 -d brady_aspec_31x5d -s 100000 -k 31
