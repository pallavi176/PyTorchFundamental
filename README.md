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

## RNN:
https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

## LSTM:
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

## GRU:
https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

