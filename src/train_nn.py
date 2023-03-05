import os
from os.path import join, dirname
import sys
sys.path.append((os.getcwd()))
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from typing import List
from src.model import LinearNet, train_one_epoch
import src.utils
import logging
from tqdm import tqdm
import time

from sklearn.metrics import accuracy_score, f1_score
logging.basicConfig(level=logging.DEBUG)


EPOCHS = 100
lr = 1e-2



PATH_DATA = join(os.getcwd(),'data')
PATH_MODELS = join(os.getcwd(),'models')

train = pd.read_csv(join(PATH_DATA, 'train_hate.csv'))
train = src.utils.create_balance_sample(train, 'labels', 200)
train.to_csv(join(PATH_DATA, 'train_subsample_active_learning.csv'))

val = pd.read_csv(join(PATH_DATA, 'val_hate.csv'))


logging.info('loaded train and validation sets with {} and {} texts respectively'.format(train.shape[0], val.shape[0]))


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

train_loader = DataLoader(TweetDataset(train.text.values.tolist(), train.labels.values.tolist()), batch_size=8)


model = LinearNet(384, 100)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

accs = []
losses = []

logging.info('Start training')

start = time.time()
for epoch in (range(EPOCHS)):
    losses.append(train_one_epoch(model, train_loader, optimizer))
end = time.time()

logging.info('End training after {} s'.format(end-start))


preds = model.predict(val.text.values.tolist()).detach().numpy()

logging.info('the model has an accuracy of {} and an f1 of {} in the val set'.format(accuracy_score(val.labels.values, preds), f1_score(val.labels.values, preds)))

torch.save(model.state_dict(), join(PATH_MODELS, 'nn_active_learn.pt'))
