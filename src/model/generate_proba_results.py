from joblib_progress import joblib_progress
from joblib import Parallel, delayed
import sys
sys.path.append('src/')
from model.proba_model import get_past_meta, predict
from model.evaluation import get_classification_report
from math import isnan
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from sklearn.metrics import classification_report
import warnings 
from model.evaluation import create_results_df
import numpy as np
warnings.filterwarnings("ignore")
import seaborn as sns


# ### Definitions
path_data_dir = 'data/'
PATH_REPORTS = 'reports/'

list_assets = ["PETR3.SA","PRIO3.SA", "VALE3.SA", "GGBR3.SA", "ABCB4.SA", "ITUB3.SA", "FLRY3.SA", "RADL3.SA"]

relevant_cols = ['Date', 'Close', 'Volume']

seq_len_list = [i for i in range(1,71)]
windows = [7,14,21]

# reverse list just for bigger n_prev_meta go first in the process
seq_len_list.reverse()

def process_combination(n_prev_meta, asset, window):
    
    meta_column = f'meta_{window}'
    
    path_results = f"{PATH_REPORTS}test_results/Proba_model_{asset.replace('.', '_')}_features={meta_column}__label={meta_column}__sql_len={n_prev_meta}_test_results.csv"
   
    # get train contigency table 
    cont_tbl_train = pd.read_csv(path_data_dir + f"processed/train_contingency_table_price_history_{asset.replace('.', '_')}_{meta_column}_range({n_prev_meta})_dataset_ffill.csv", index_col=0, header=[i for i in range(n_prev_meta)])
    
    # create probability table
    df_probas = cont_tbl_train.apply(lambda x: [col/sum(x) for col in x])    
    
    # get test dataset
    test_dataset = pd.read_csv(path_data_dir + f"processed/test_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
    
    # get the past meta for all days    
    test_dataset["past_meta"] = test_dataset.apply(lambda x: get_past_meta(test_dataset,x,n_prev_meta,name_meta_col = meta_column), axis=1)

    # # value to fill firsts days, that do not have input sequence.
    fill_value = tuple("-1000" for n in range(n_prev_meta)) if n_prev_meta != 1 else -1000
    
    # remove first rows
    test_dataset.iloc[:n_prev_meta, -1] = fill_value
    
    # get y_test
    y_test = test_dataset[meta_column]
    
    # predict based on probability table
    y_pred = test_dataset.past_meta.apply(lambda x: predict(x, df_probas))
    
    # when the value is nan, repeat the last predict
    y_pred.fillna(-1000, inplace = True)
    
    # Create and save the results dataframe
    results_df = create_results_df(y_test, y_pred)
    # print(f"Results saved to: {path_results}")
    results_df.to_csv(path_results, index=False)


total_combinations = len(seq_len_list)*len(windows)*len(list_assets)

with joblib_progress("Calculating square...", total=total_combinations):
    Parallel(n_jobs=-1)(delayed(process_combination)(
        n_prev_meta, asset, window) 
                       for n_prev_meta in seq_len_list
                       for window in windows
                       for asset in list_assets
                       )