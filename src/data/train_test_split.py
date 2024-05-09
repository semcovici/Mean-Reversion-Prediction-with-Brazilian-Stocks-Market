

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from math import isnan
import matplotlib.pyplot as plt

path_data_dir = 'data/'

list_assets = ["PETR3.SA","PRIO3.SA", "VALE3.SA", "GGBR3.SA", "ABCB4.SA", "ITUB3.SA", "FLRY3.SA", "RADL3.SA"]

relevant_cols = ['Date', 'Close', 'Volume']

# simple moving average 
window_size = 21

train_size = 0.3

n_prev_meta = 10

from data_split import temporal_train_test_split





def main():
    
    for asset in tqdm(list_assets):
        
        data = pd.read_csv(path_data_dir + f"processed/price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
        
        train_dataset, test_dataset = temporal_train_test_split(
            df = data
        )
        
        
        train_dataset.to_csv(path_data_dir + f"processed/train_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
        test_dataset.to_csv(path_data_dir + f"processed/test_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
        
        
if __name__ == '__main__':
    
    main()
    