import os
from os.path import join, dirname
import sys
sys.path.append((os.getcwd()))
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from typing import List
from src.model import FCSTNet, train_one_epoch
import src.utils
import logging
from tqdm import tqdm
import time
import utils

from sklearn.metrics import accuracy_score, f1_score
logging.basicConfig(level=logging.DEBUG)


EPOCHS = 2
lr = 1e-2
n = 100

class TweetDataset(Dataset):
    def __init__(self, texts: List, labels: List):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)
		
    def __getitem__(self, idx):
        texts = self.texts[idx]
        labels = self.labels[idx]
        return texts, torch.tensor(labels, dtype=torch.long)


PATH_DATA = join(os.getcwd(),'data')
PATH_MODELS = join(os.getcwd(),'models')

val = pd.read_csv(join(PATH_DATA, 'val_hate.csv'))
train = pd.read_csv(join(PATH_DATA, 'train_hate.csv'))

for seed in range(4):

    train = src.utils.create_balance_sample(train, 'labels', 200)
    print(train)
    train_loader = DataLoader(TweetDataset(train.text.values.tolist(), train.labels.values.tolist()), batch_size=8)


    model = FCSTNet(384)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in (range(EPOCHS)):
        losses.append(train_one_epoch(model, train_loader, optimizer))

    try:
        torch.save(model.state_dict(), join(PATH_MODELS, f'st_standard_hate_{n}',f'stfc_{seed}.pt'))
    except:
        os.mkdir(join(PATH_MODELS, f'st_standard_hate_{n}'))
        torch.save(model.state_dict(), join(PATH_MODELS, f'st_standard_hate_{n}',f'stfc_{seed}.pt'))
