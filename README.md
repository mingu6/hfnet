A fork of https://github.com/ethz-asl/hfnet. See original repo for more information, and refer to the below works.

# hfnet
From Coarse to Fine: Robust Hierarchical Localization at Large Scale with HF-Net (https://arxiv.org/abs/1812.03506)

# Setup

First, create a conda environment with the basic packages using the following command

`conda create -n hfnet tensorflow-gpu=1.12.0 cudatoolkit=9.0 pytorch=0.4.1`

Next, install the package and dependencies (cd to base directory)

`make install`

You will be prompted to input the following directories after running make:

- Data directory: Visual localization challenge base directories
- Experiments directory: Stores hfnet weights and model outputs (e.g. extracted features and localization results)
- Raw directory: This directory will be created and raw RobotCar INS data will be downloaded here to run experiments

We will store our custom code for our experiments in the `QUT` directory within the base dir. Thirdparty repos like the RobotCar datascraper and the SDK will be stored inside `Thirdparty`. 

# Data

For the datasets, setup the directory structure as per the original HF-Net repo documentation (or come see me for the hard drive).

# Extract features

The first step after setting up your virtual environment and the dataset is to extract local and global features from the reference and query datasets. First, run

`hfnet/export_predictions.py \                
        hfnet/configs/hfnet_export_robotcar_db.yaml \
        robotcar \
        --exper_name hfnet \
        --keys keypoints,scores,local_descriptors,global_descriptor`

`hfnet/export_predictions.py \                
        hfnet/configs/hfnet_export_robotcar_queries.yaml \
        robotcar \
        --exper_name hfnet \
        --keys keypoints,scores,local_descriptors,global_descriptor`

This will take up ~400Gb of hard drive space.

# Localize

To localize, run the following for the first localization

`python3 hfnet/evaluate_robotcar.py \                 
        hfnet_model \
        robotcar_[night|sun|night-rain|etc...] \
        --local_method hfnet \
        --global_method hfnet \
        --build_db \
        --queries night \
        --export_poses`

For subsequent localizations remove the --build_db flag as this only needs to be done once. I would recommend setting a swap of 64Gb (can confirm that this works) with 32Gb RAM to run successfully.
