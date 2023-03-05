from numpy import concatenate
from datasets import load_dataset
import datasets
import os 
from os.path import join
import sys
import utils

import pandas as pd



PATH_DATA = join(os.getcwd(), 'data')

train = utils.load_and_subsample(join(PATH_DATA, 'train_hate'), 'labels', 8)
print(train)

data = pd.read_csv(join(PATH_DATA, 'train_hate.csv'))
print(data.head(5))

print(utils.create_balance_sample(data, 'labels', 8))






