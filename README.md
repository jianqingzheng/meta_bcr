<div align="center">
<h1> Meta-BCR </h1>

<a href="https://jianqingzheng.github.io/meta_bcr/"><img alt="Website" src="https://img.shields.io/website?url=https%3A%2F%2Fjianqingzheng.github.io%2Fmeta_bcr%2F&up_message=online&up_color=darkcyan&down_message=offline&down_color=darkgray&label=Project%20Page"></a>
[![arXiv](https://img.shields.io/badge/arXiv-xxx-b31b1b.svg)]()
[![Explore MetaBCR in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jianqingzheng/meta_bcr/blob/main/meta_bcr.ipynb)

</div>


Code for paper [Meta-BCR unveils a germinal center-derived atypical memory B cell subset expressing broadly neutralizing antibodies]()


> This repo provides an implementation of the training and inference pipeline of **Meta-BCR** based on Pytorch. 

---
### Contents ###
- [0. Brief Introduction](#0-brief-intro)
- [1. Installation](#1-installation)
- [2. Usage](#2-usage)
  - [2.1. Setup](#21-setup)
  - [2.2. Training](#22-training)
  - [2.3. Inference](#23-inference)
- [3. Demo](#3-demo)
- [4. Citing this work](#4-citing-this-work)


---

## 0. Brief Intro ##

![image](https://github.com/user-attachments/assets/36ffb983-1eb8-4c0d-8456-39dc569e8a23)


The identification of broadly protective memory B cells is crucial for developing vaccines and antibody therapies, yet the task remains challenging due to its rarity. Meta-BCR, a computational framework integrating meta-learning and the Mean Teacher model, is developed by using single-cell V(D)J sequencing data but without the need for antigen labelling, which is efficient in predicting B-cell receptor (BCR) functionality and identifying rare B cell subsets. Meta-BCR enables the discovery of a conserved subset of germinal centre (GC)-derived atypical memory B cells enriched in broadly neutralizing clonotypes across influenza, SARS-CoV-2, and RSV datasets. These findings highlight key somatic hypermutations (SHMs) associated with enhanced cross-neutralization against viral variants, supported by cryo-EM studies demonstrating that these mutations enhance hydrophobic interactions to promote broader neutralization. Overall, Meta-BCR identifies a unique atypical memory B cell with broad protective potential in the infectious diseases, which may guide the design of effective antiviral strategies.

---
## 1. Installation ##

Clone code from Github repo: https://github.com/jianqingzheng/meta_bcr.git
```shell
git clone https://github.com/jianqingzheng/meta_bcr.git
cd meta_bcr/
```


install packages

[![OS](https://img.shields.io/badge/OS-Windows%7CLinux-darkblue)]()
[![PyPI pyversions](https://img.shields.io/badge/Python-3.8-blue)](https://pypi.python.org/pypi/ansicolortags/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1+cu113-lightblue)](https://pytorch.org/)
[![Numpy](https://img.shields.io/badge/Numpy-1.19.5-lightblue)](https://numpy.org)

This project only requires basic packages such as transformer, torch, pandas, etc. Use Anaconda to manage the packages is recommended.

```shell
conda create -n metabcr python=3.10
# See torch installation on https://pytorch.org/get-started/locally/
conda install conda-forge::transformers
```

> Other versions of the packages could also be applicable



---
## 2. Usage ##

### 2.1. Setup ###

Directory layout:
```
[$DOWNLOAD_DIR]/meta_bcr/
├── Analysis/
|   └── ...
├── Config/
|   |   # configure file (.json files)
|   └── config_[$data_name].json
|   └── ...
├── Data/
|   ├── /
|   └── ...
├── External/
|   ├── prot_bert/
|   └── ...
├── MetaBCR/
|   ├── /
|   └── ...
├── Models/
|   └── ...
└── ...
```

### 2.2. Training ###

To get started, please download the necessary data and pretrained models from the following Google Drive links:

- **The training and testing data:** Download the contents of the ```Data/``` folder from [this link](https://drive.google.com/drive/folders/1E8jZun1-iUpO8jkWVriW07B4tCZx_BUM?usp=sharing).

- **Pretrained BERT model:** Download the contents of the ```External/``` folder from [this link](https://drive.google.com/drive/folders/10Qoqy0zcM3L7knLG1KGOlUOmXMa2SrLp?usp=sharing).

- **Pretrained MetaBCR models:** Download the following from [this link](https://drive.google.com/drive/folders/1CAQjVVd8SpRdG7xkr4aSMyVNOXeT6Xbn?usp=sharing):
  - Pretrained MataBCR model in ```Models/240612-flu-bind``` 
  - The semi-supervised fine-tuned model in ```Models/240822-flu-bind``` 

Once all files are in place, you can run the semi-supervised training on the *flu-bind* dataset using:

```bash
conda activate metabcr
python train_semi_supervise.py --dataset flu-bind
```

---
## 3. Demo ##

You can test flu bind via:

```bash
python test_single_cell.py
```

---

## 4. Citing this work

Any publication that discloses findings arising from using this source code or the network model should cite:

```bibtex

```
