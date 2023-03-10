import os
import pandas as pd

from os import getcwd
from os.path import join, dirname

from datetime import datetime
import json
from tqdm import tqdm
import numpy as np 
PATH_REPO = os.getcwd()
PATH_DATA = join(PATH_REPO, 'data')
PATH_UTILS = join(PATH_REPO, 'utils')
import sys
sys.path.append((os.getcwd()))
sys.path.append(PATH_UTILS)
import src.utils

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPTNeoModel

N = 4
pd.set_option('display.max_colwidth', None)


train = pd.read_csv(join(PATH_DATA, 'train_hate.csv'))
val = pd.read_csv(join(PATH_DATA, 'val_hate.csv'))
mapping = {0: 'NO', 1: 'YES'}

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

prompt = "I want you to classify tweets, wether they contain hate or not. \n \
 Here some training examples: \n "


preds = {}
for seed in np.arange(4):
    answers = []

    train = src.utils.create_balance_sample(train, 'labels', N, seed = seed)
    prompt_train = prompt

    for text, label in zip(train.text.values, train.labels.values):
        prompt_train +=  f'Tweet: {text}'  + f' \n Label: {mapping[label]} \n \n'  

    for k,tweet in tqdm(enumerate(val.text.values)):
        prompt_final = prompt_train + f'\n Now predict the following \n Tweet: {tweet} \n Label: '

        inputs = tokenizer(prompt_final, return_tensors="pt")
        outputs = model.generate(**inputs)
        answers.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    preds[f'seed_{str(seed)}'] = answers
    print(preds)

with open(join(PATH_DATA,f"preds_prompt_llm_{str(N)}.json"), "w") as outfile:
    json.dump(preds, outfile)