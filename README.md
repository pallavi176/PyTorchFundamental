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


## Deploy a PyTorch model using Flask & Heroku

Create and Deploy your first Deep Learning app! In this PyTorch tutorial we learn how to deploy our PyTorch model with Flask and Heroku.
We create a simple Flask app with a REST API that returns the result as json data, and then we deploy it to Heroku. As an example PytTorch app we do digit classification, and at the end I show you how I draw my own digits and then predict it with our live running app.


## Contextual Chatbots with Tensorflow
https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

### Implementation of a Contextual Chatbot in PyTorch.  
Simple chatbot implementation with PyTorch. 

- The implementation should be easy to follow for beginners and provide a basic understanding of chatbots.
- The implementation is straightforward with a Feed Forward Neural net with 2 hidden layers.
- Customization for your own use case is super easy. Just modify `intents.json` with possible patterns and responses and re-run the training (see below for more info).

### Installation:
 ```console
pip install nltk
 ```

If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

### Usage
Run
```console
python train.py
```
This will dump `data.pth` file. And then run
```console
python chat.py
```

### Customize
Have a look at [intents.json](intents.json). You can customize it according to your own use case. Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. You have to re-run the training whenever this file is modified.
```console
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
    },
    ...
  ]
}
```

## Custom Dataset

### SpaCy Installation
``` bash
python -m spacy download en_core_web_sm
```

### Flickr8k Dataset
https://www.kaggle.com/datasets/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb


## Image Captioning

Download the dataset used: https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb
Then set images folder, captions.txt inside a folder Flickr8k.

train.py: For training the network

model.py: creating the encoderCNN, decoderRNN and hooking them togethor

get_loader.py: Loading the data, creating vocabulary

utils.py: Load model, save model, printing few test cases downloaded online

utils.py => get_loader.py => model.py => train.py

