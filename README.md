<div align="center">
<h1> Meta-BCR </h1>

<a href="https://jianqingzheng.github.io/meta_bcr/"><img alt="Website" src="https://img.shields.io/website?url=https%3A%2F%2Fjianqingzheng.github.io%2Fmeta_bcr%2F&up_message=online&up_color=darkcyan&down_message=offline&down_color=darkgray&label=Project%20Page"></a>
[![arXiv](https://img.shields.io/badge/arXiv-xxx-b31b1b.svg)]()
[![Explore MetaBCR in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jianqingzheng/meta_bcr/blob/main/meta_bcr.ipynb)

</div>

<table>
  <tr>
    <td><img src="docs/static/images/demo_1.gif" alt="demo_fig1" width="100%" /></td>
    <td><img src="docs/static/images/demo_2.gif" alt="demo_fig2" width="100%" /></td>
  </tr>
</table>

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

![header](docs/static/images/graphic_abstract.png)
The identification of broadly protective memory B cells is crucial for developing vaccines and antibody therapies, yet the task remains challenging due to its rarity. Meta-BCR, a computational framework integrating meta-learning and the Mean Teacher model, is developed by using single-cell V(D)J sequencing data but without the need for antigen labelling, which is efficient in predicting B-cell receptor (BCR) functionality and identifying rare B cell subsets. Meta-BCR enables the discovery of a conserved subset of germinal centre (GC)-derived atypical memory B cells enriched in broadly neutralizing clonotypes across influenza, SARS-CoV-2, and RSV datasets. These findings highlight key somatic hypermutations (SHMs) associated with enhanced cross-neutralization against viral variants, supported by cryo-EM studies demonstrating that these mutations enhance hydrophobic interactions to promote broader neutralization. Overall, Meta-BCR identifies a unique atypical memory B cell with broad protective potential in the infectious diseases, which may guide the design of effective antiviral strategies.


Highlight:
<ul style="width: auto; height: 200px; overflow: auto; padding:0.4em; margin:0em; text-align:justify; font-size:small">
  <li> <b>Point1</b>: xxx;
  </li>
  <li> <b>Point2</b>: xxx.
  </li>
</ul>

---
## 1. Installation ##

Clone code from Github repo: https://github.com/jianqingzheng/meta_bcr.git
```shell
git clone https://github.com/jianqingzheng/meta_bcr.git
cd meta_bcr/
```


install packages

This project only requires basic packages such as transformer, torch, pandas, etc. Use Anaconda to manage the packages is recommended.

[![OS](https://img.shields.io/badge/OS-Windows%7CLinux-darkblue)]()
[![PyPI pyversions](https://img.shields.io/badge/Python-3.8-blue)](https://pypi.python.org/pypi/ansicolortags/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1+cu113-lightblue)](https://pytorch.org/)
[![Numpy](https://img.shields.io/badge/Numpy-1.19.5-lightblue)](https://numpy.org)

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
├── External/
|   └── ...
└── ...
```

Configuration setting:

<div align="center">
	
| Argument              | Example           | Description                                	|
| --------------------- | ----------------- |----------------------------------------------|
| `--data_name` 	    |'flu', 'rbd'        | The data  name                    |
| `--net_name` 	        |'acnn'             | The network name                    |
| `--ndims` 	        |2, 3                | The dimension of image                    |
</div>

> configuration settings are edited in `[$DOWNLOAD_DIR]/meta_bcr/Config/*.yaml`


### 2.2. Training ###

Download the data and pretrained models [here](https://drive.google.com/drive/folders/1om6Rt9kvjuebvVd3TrouVkCuTKVWYAjX?usp=sharing).

You can run flu-bind training via:

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

A demo can be found in the provided [notebook](https://github.com/jianqingzheng/meta_bcr/blob/main/meta_bcr.ipynb).

Alternatively, it can be easily run via [![Explore Meta-BCR in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jianqingzheng/meta_bcr/blob/main/meta_bcr.ipynb).


---

## 4. Citing this work

Any publication that discloses findings arising from using this source code or the network model should cite:

```bibtex

```
