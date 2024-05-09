
# ### Imports


import plotly.graph_objects as go

import pandas as pd
from tqdm import tqdm

# ### Definitions
path_data_dir = 'data/'

list_assets = ["PETR3.SA","PRIO3.SA", "VALE3.SA", "GGBR3.SA", "ABCB4.SA", "ITUB3.SA", "FLRY3.SA", "RADL3.SA"]

relevant_cols = ['Date', 'Close', 'Volume']


def main():
    
    for n_prev_meta in range(1, 21):
        
        for asset in tqdm(list_assets, desc=f'{n_prev_meta} of {10}'):
            
            # create for full dataset
            data = pd.read_csv(path_data_dir + f"processed/price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
            
            cont_tbl = create_contigency_table(data,n_prev_meta, progress_bar=False)
            
            cont_tbl.to_csv(path_data_dir + f"processed/contingency_table_price_history_{asset.replace('.', '_')}_meta_range({n_prev_meta})_dataset_ffill.csv")
            
            # create for train test
            train = pd.read_csv(path_data_dir + f"processed/train_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
            test = pd.read_csv(path_data_dir + f"processed/test_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
            
            cont_tbl_train = create_contigency_table(train,n_prev_meta, progress_bar=False)
            cont_tbl_test = create_contigency_table(test,n_prev_meta, progress_bar=False)
            
            cont_tbl_train.to_csv(path_data_dir + f"processed/train_contingency_table_price_history_{asset.replace('.', '_')}_meta_range({n_prev_meta})_dataset_ffill.csv")
            cont_tbl_test.to_csv(path_data_dir + f"processed/test_contingency_table_price_history_{asset.replace('.', '_')}_meta_range({n_prev_meta})_dataset_ffill.csv")


if __name__ == '__main__':
    
    main()
