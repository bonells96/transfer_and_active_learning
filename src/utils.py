import pandas as pd
import numpy as np
from datasets import Dataset

def load_and_subsample(name_file:str, target_column:str, n:int, seed:int=13) -> pd.DataFrame:
    data = pd.read_csv(f'{name_file}.csv')
    sub_data = create_balance_sample(data, target_column, n, seed)
    return Dataset.from_pandas(sub_data)
    

def create_balance_sample(data: pd.DataFrame, target_column:str, n:int, seed:int=13) -> pd.DataFrame:
    "returns a balanced subsample"
    sample_ = pd.DataFrame()
    unique_values = np.unique(data.loc[:,target_column].values)
    for value in unique_values:
        m = min(data.loc[data.loc[:,target_column]==value,].shape[0], n//len(unique_values))
        sample_ = pd.concat((sample_, data.loc[data.loc[:,target_column]==value,].sample(m, random_state=seed)))
    return sample_.sample(n)



