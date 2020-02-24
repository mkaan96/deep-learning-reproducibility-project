# Deep Learning Reproducibility Project

## Requirements
- Miniconda (https://docs.conda.io/en/latest/miniconda.html)
- Data folder with some samples (~155 MB) (https://drive.google.com/open?id=1c7-kEHPj4eoZUXwjzDOcCS9C_EIVz3gT)
- (Optional) Data folder with all samples (~11 GB) (https://drive.google.com/open?id=1Gd09zytoGK_vhqLpPIUGiHccjRnA1IbB)

## Installation
- Download and install miniconda
- Add miniconda3/bin to your PATH
  - Verify with the following command in your terminal: `conda -V`
- Run the following command in your terminal from the root of this project:

`conda env create -f environment.yml`

- Download and extract the data archive in the root of the project
  - Verify that the `data` folder exists

## Training
Run the following command in the terminal from the root of this project:

```
source activate deep-learning-project
python 03_train_gcnn.py --gpu -1
```

or for the big dataset:

```
source activate deep-learning-project
python 03_train_gcnn.py --gpu -1 --problem setcover
```

## Testing
Run the following command in the terminal from the root of this project:

```
source activate deep-learning-project
python 04_train_gcnn.py --gpu -1
```

or for the big dataset:

```
source activate deep-learning-project
python 04_train_gcnn.py --gpu -1 --problem setcover
```