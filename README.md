# PyTorch Fundamentals
PyTorch Fundamentals

# STEPS:

## STEP 1: create environment
conda create --prefix ./env python=3.8 -y

## STEP 2:  activate environment
source activate ./env

## STEP 3: Install Pytorch
Go to website: https://pytorch.org/
Select OS: Windows
Package: Conda
Language: Python
Compute Platform: CPU

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## STEP 3: Install ipykernel to run notebook

```bash
pip install ipykernel
```

## STEP 4: Install requirements

```bash
pip install -r requirements.txt
```

### Download hymenoptera data
data_URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

