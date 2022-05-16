# A Contrastive Objective for Disentangled Representations

### DCoDR
> [A Contrastive Objective for Disentangled Representations](https://arxiv.org/abs/2203.11284) \
> Joanthan Kahana and Yedid Hoshen \
> Official PyTorch Implementation

> **Abstract:** Learning representations of images that are invariant to 
> sensitive or unwanted attributes is important for many tasks including bias 
> removal and cross domain retrieval. Here, our objective is to learn 
> representations that are invariant to the domain (sensitive attribute) 
> for which labels are provided, while being informative over all other image 
> attributes, which are unlabeled. We present a new approach, proposing a new 
> domain-wise contrastive objective for ensuring invariant representations. 
> This objective crucially restricts negative image pairs to be drawn from the 
> same domain, which enforces domain invariance whereas the standard contrastive 
> objective does not. This domain-wise objective is insufficient on its own as 
> it suffers from shortcut solutions resulting in feature suppression. We overcome 
> this issue by a combination of a reconstruction constraint, image augmentations 
> and initialization with pre-trained weights. Our analysis shows that the choice 
> of augmentations is important, and that a misguided choice of augmentations can 
> harm the invariance and informativeness objectives. In an extensive evaluation, 
> our method convincingly outperforms the state-of-the-art in terms of representation 
> invariance, representation informativeness, and training speed. Furthermore,
> we find that in some cases our method can achieve excellent results even without 
> the reconstruction constraint, leading to a much faster and resource efficient training.


This repository is the official PyTorch implementation of [A Contrastive Objective for Disentangled Representations](https://arxiv.org/abs/2203.11284)

<a href="https://arxiv.org/abs/2203.11284" target="_blank"><img src="https://img.shields.io/badge/arXiv-2203.11284-b31b1b.svg"></a>

## Usage

By default the `<base-dir>` directory is the main directory of the repository, although it can be changed in the code itself. 

### Requirements
![python >= 3.7.3](https://img.shields.io/badge/python->=3.7.3-blue.svg) 
![cuda >= 11.1](https://img.shields.io/badge/CUDA->=11.1-bluegreen.svg) 
![pytorch >= 1.9.0](https://img.shields.io/badge/pytorch->=1.9.0-orange.svg)

* ![numpy >= 1.16.2](https://img.shields.io/badge/numpy->=1.16.2-purple.svg)
* ![pandas >= 1.3.4](https://img.shields.io/badge/pandas->=1.3.4-darkblue.svg)
* ![faiss_gpu >= 1.7.2](https://img.shields.io/badge/faiss_gpu->=1.7.2-darkgreen.svg)
* ![face_alignment >= 1.3.5](https://img.shields.io/badge/face_alignment->=1.3.5-yellow.svg)
* ![dlib >= 19.17.0](https://img.shields.io/badge/dlib->=19.17.0-red.svg)


### Downloading The ImageNet Pre-Trained Weights

Please create a directory `<base-dir>/pretrained_weights` and put the ImageNet pre-trained weights in it.

MocoV2 weights can be downloaded from [here](https://drive.google.com/drive/folders/1SAsU6OQz38TXkzRcBpUxianEK35g6UHy?usp=sharing) or from the official [github page](https://github.com/facebookresearch/moco)


### Downloading the Pre-Processed Datasets

**NOTE:** you need to download the datasets first. 

We provide pre-processed versions of the datasets. They are found in [here](https://drive.google.com/drive/folders/1i1rgZFBPAXnlbUsYp9Fnh5IqyQJGnjzq?usp=sharing).

Please put the pre-processed versions under `cache/preprocess`.

### Pre-Processing the Datasets Yourself (Optional)

**NOTE:** As mentions above you can download the pre-processed versions from [here](https://drive.google.com/drive/folders/1i1rgZFBPAXnlbUsYp9Fnh5IqyQJGnjzq?usp=sharing).

We also supply scripts for creating the pre-processed versions.

* Edges2Shoes can be downloaded by running the given script `scripts/downlod_e2s_zappos.sh`
* Cars3D can be downloaded by the documentation of [disentanglement_lib](https://github.com/google-research/disentanglement_lib)
* Shapes3D can be downloaded from [here](https://github.com/deepmind/3d-shapes). Please put it under `$DISENTANGLEMENT_LIB_DATA/3dshapes/`
* CelebA can be downloaded from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Please download its files under `raw_data/celeba`

In case you download the datasets to other locations, make sure to update the path in the beginning of the corresponding preprocessing script before running it.

The preprocessing can be applied by:

    scripts/prepare_#DATASET_NAME#.py

### Training

Given a preprocessed train set and test set as the scripts create, 

Training a dataset can be done by running one of the attached bash scripts in the bash_scripts folder, 
according to the desired experiment.

To train DCoDR on smallnorb for example, simply run:

    bash bash_scripts/DCoDR/smallnorb/DCoDR__smallnorb__pipeline.sh

## Trained Models

We provide trained models for all of the evaluated datasets from the main experiment in the paper.
Please download model `.pth` files as well as the `config.pkl` file which is needed for evaluation.

| Dataset     | DCoDR-norec                                                                                                        | DCoDR                                                                                                        |
|:------------|:-------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|
| Cars3D      | [DCoDR-norec Cars3d](https://drive.google.com/drive/folders/1E2ITEb8CPuQWDllfZrlyLbwaMtO1jsNB?usp=sharing)         | [DCoDR Cars3D](https://drive.google.com/drive/folders/1Lbb6hGONLrGLlgp_TS9TcJsaaF8ri-lb?usp=sharing)         |
| SmallNorb   | [DCoDR-norec SmallNorb](https://drive.google.com/drive/folders/13GtahDl4ZmbGIGJUeZzhyX7mEusrImB7?usp=sharing)      | [DCoDR SmallNorb](https://drive.google.com/drive/folders/1_whq18GzVarMPxMyTnWwKzp6I1SaBUFh?usp=sharing)      |
| CelebA      | [DCoDR-norec CelebA](https://drive.google.com/drive/folders/10kebgAzg9aF-roVeQ7kE0fMAVulymPqB?usp=sharing)         | [DCoDR CelebA](https://drive.google.com/drive/folders/1tLT_AYKG2c3PpZnERGhst122mGrvBc8J?usp=sharing)         |
| Edges2Shoes | [DCoDR-norec Edges2Shoes](https://drive.google.com/drive/folders/10N45y5Y5lWj0sIhyJYu_B70GmoEwxxSX?usp=sharing)    | [DCoDR Edges2Shoes](https://drive.google.com/drive/folders/1oFCM0AVuRhB5wewpMfL6mNxqV3lgY8jd?usp=sharing)    |
| Shapes3D    | [DCoDR-norec Shapes3D](https://drive.google.com/drive/folders/1b6N9cPFwif24YD5j04WqgM_VteHUlJB5?usp=sharing)       | [DCoDR Shapes3D](https://drive.google.com/drive/folders/1UaGcoeEWRnyT1O_D615eB6Qzl6VS2IJC?usp=sharing)       |


## Citation
If you find this useful, please cite our paper:
```
@article{kahana2022dcodr,
  title={A Contrastive Objective for Disentangled Representations},
  author={Kahana, Jonathan and Hoshen, Yedid},
  journal={arXiv preprint arXiv:2203.11284},
  year={2022}
}
```

