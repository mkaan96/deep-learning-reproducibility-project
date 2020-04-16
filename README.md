# Deep Learning Reproducibility Project
By Kaan Yilmaz, Beyza Hizli and Manisha Sethia

## Introduction
This repository contains the source code we used to reproduce GCNN model for setcover instances, extra hyperparameter 
search and an evaluation of an additional dataset (mik) of the paper:

Exact Combinatorial Optimization with Graph Convolutional Neural Networks by Gasse et al.

## Requirements
- Miniconda (https://docs.conda.io/en/latest/miniconda.html)
- (Optional) Data folder with some setcover samples (~155 MB) (https://drive.google.com/open?id=1c7-kEHPj4eoZUXwjzDOcCS9C_EIVz3gT)
- Data folder with all setcover samples (~11 GB) (https://drive.google.com/open?id=1Gd09zytoGK_vhqLpPIUGiHccjRnA1IbB)
- Extra data folder with all mik samples (~3 GB) (https://drive.google.com/open?id=1KZeRtykYhHDUpVW4N1F-Nq_BsqwXX0cJ)

## Installation
- Download and install miniconda
- Add miniconda3/bin to your PATH
  - Verify with the following command in your terminal: `conda -V`
- Run the following command in your terminal from the root of this project:

`conda env create -f environment.yml`

- Download and extract the data archive in the root of the project
  - Verify that the `data` folder exists
  - The folder structure for setcover should be `data/samples/setcover/500r_1000c_0.05d/{train|test|valid}/*.pkl`
  - The folder structure for mik should be `data/samples/mik/{train|test|valid}/*.pkl`

## Training
We have provided the trained models ourselves, see the folder `trained_models`. If you want to train it anyway, then please rename this folder to something else.
### Original Setcover instances
Run the following command in the terminal from the root of this project:

```
source activate deep-learning-project
python train_gcnn.py --seed 0
python train_gcnn.py --seed 1
python train_gcnn.py --seed 2
python train_gcnn.py --seed 3
python train_gcnn.py --seed 4
```

### Extra hyperparameter search
Run the following command in the terminal from the root of this project:

```
source activate deep-learning-project
python train_gcnn.py --lr 0.0001  
python train_gcnn.py --lr 0.01  
python train_gcnn.py --optimizer RMSprop
```

### Additional dataset
Run the following command in the terminal from the root of this project:

```
source activate deep-learning-project
python train_gcnn.py --problem mik --samples_path data/samples/mik
```

### Results
The log and trained model parameters are stored in:

`trained_models/{problem}/baseline/{seed}/{lr-high|lr-low|lr-normal}/{optimizer}/`

## Testing
### Original Setcover instances
Run the following command in the terminal from the root of this project:

```
source activate deep-learning-project
python test.py
```

### Extra hyperparameter search
Run the following command in the terminal from the root of this project:

```
source activate deep-learning-project
python test.py --lr 0.001  
python test.py --lr 0.1  
python test.py --optimizer RMSprop
```

### Additional dataset
```
source activate deep-learning-project
python test.py --problem mik --samples_path data/samples/mik
```

### Results
The results are stored in the folder `results/{problem}_test_{date}.csv`

## Help: running out of (GPU) memory
Try to reduce the value of `valid_batch_size` in `config.json`
