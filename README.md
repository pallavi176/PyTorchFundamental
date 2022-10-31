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

## Run tensorboard

```bash
cd notebook
tensorboard --logdir=runs
```

### Go to link
http://localhost:6006/

## RNN Data:
Download the data: https://download.pytorch.org/tutorial/data.zip

### Further Readings:

https://karpathy.github.io/2015/05/21/rnn-effectiveness/
https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks#architecture
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

### RNN:
https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

### LSTM:
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

### GRU:
https://pytorch.org/docs/stable/generated/torch.nn.GRU.html


## Lightning Repo
https://github.com/PyTorchLightning/pytorch-lightning

### Installation
```bash
conda install pytorch-lightning -c conda-forge
or
pip install pytorch-lightning
```

### Run script
```bash
cd src/pytorch-lightning/
python lightning.py
tensorboard --logdir=lightning_logs
```
#### To view tensorboard:
http://localhost:6006/


## PyTorch LR Scheduler 
https://pytorch.org/docs/stable/optim.html


## ## Deploy a PyTorch model using Flask & Heroku

Create and Deploy your first Deep Learning app! In this PyTorch tutorial we learn how to deploy our PyTorch model with Flask and Heroku.
We create a simple Flask app with a REST API that returns the result as json data, and then we deploy it to Heroku. As an example PytTorch app we do digit classification, and at the end I show you how I draw my own digits and then predict it with our live running app.

