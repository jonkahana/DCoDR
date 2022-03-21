# A Contrastive Objective for Disentangled Representations

### DCoDR
> A Contrastive Objective for Disentangled Representations \
> Joanthan Kahana and Yedid Hoshen \
> Official PyTorch Implementation

## Usage
### Dependencies
* python >= 3.7.3
* numpy >= 1.16.2
* pytorch >= 1.9.0
* dlib >= 19.17.0
* face-alignment >= 1.3.5
* faiss-gpu >= 1.7.2

### Downloading the Datasets

We already pre-processed the smallnorb dataset as an example, which is found in:
    cache/preprocess/smallnorb_train.sh, cache/preprocess/smallnorb_test.sh

For the others we supply a script for each one which preprocesses.

NOTE: you need to download the datasets first. 

Edges2Shoes can be downloaded by running the given script: 

    bash scripts/downlod_e2s_zappos.sh

Cars3D can be downloaded by the documentation in disentanglement_lib library: https://github.com/google-research/disentanglement_lib

Shapes3D can be downloaded from: https://github.com/deepmind/3d-shapes, to $DISENTANGLEMENT_LIB_DATA/3dshapes/

CelebA can be downloaded from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, to raw_data/celeba

In case you download the datasets to other locations, make sure to update the path in the beginning of the corresponding preprocessing script before running it.

### Preprocessing The Datasets

The preprocessing can be applied by:

    scripts/prepare_#DATASET_NAME#.py

### Training

Given a preprocessed train set and test set as the scripts create, 

Training a dataset can be done by running one of the attached bash scripts in the bash_scripts folder, 
according to the desired experiment.

To train DCoDR on smallnorb for example, simply run:

    bash bash_scripts/DCoDR/smallnorb/DCoDR__smallnorb__pipeline.sh
