
# # ### Imports


# import plotly.graph_objects as go

# import pandas as pd
# from tqdm import tqdm


# from contigency_table import create_contigency_table
# # ### Definitions
# path_data_dir = 'data/'

# list_assets = ["PETR3.SA","PRIO3.SA", "VALE3.SA", "GGBR3.SA", "ABCB4.SA", "ITUB3.SA", "FLRY3.SA", "RADL3.SA"]

# relevant_cols = ['Date', 'Close', 'Volume']

# windows = [7,14,21]
# max_seq_len = 70

# def main():
    
    
#     for n_prev_meta in range(1, max_seq_len):
        
#         for asset in tqdm(list_assets, desc=f'{n_prev_meta} of {max_seq_len}'):
            
#             for window in windows:
            
            
#                 data = pd.read_csv(path_data_dir + f"processed/price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
                
#                 train = pd.read_csv(path_data_dir + f"processed/train_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
#                 test = pd.read_csv(path_data_dir + f"processed/test_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
                
#                 ######################
#                 # dependent
#                 ######################
                
#                 column = f'meta_{window}'
                
#                 # create for full dataset
#                 # cont_tbl = create_contigency_table(data,n_prev_meta, col=column, progress_bar=False)
#                 # cont_tbl.to_csv(path_data_dir + f"processed/contingency_table_price_history_{asset.replace('.', '_')}_meta_range({n_prev_meta})_dataset_ffill.csv")
                
#                 # create for train test
#                 cont_tbl_train = create_contigency_table(train,n_prev_meta, column, progress_bar=False)
#                 cont_tbl_test = create_contigency_table(test,n_prev_meta, column, progress_bar=False)
                
#                 cont_tbl_train.to_csv(path_data_dir + f"processed/train_contingency_table_price_history_{asset.replace('.', '_')}_{column}_range({n_prev_meta})_dataset_ffill.csv")
#                 cont_tbl_test.to_csv(path_data_dir + f"processed/test_contingency_table_price_history_{asset.replace('.', '_')}_{column}_range({n_prev_meta})_dataset_ffill.csv")


#                 ######################
#                 # independent
#                 ######################
                
#                 # this models was discarted
                
#                 # # create for full dataset
#                 # cont_tbl = create_contigency_table(data,n_prev_meta, progress_bar=False, return_only_n_prev_meta=True)
                
#                 # cont_tbl.to_csv(path_data_dir + f"processed/contingency_table_ind_price_history_{asset.replace('.', '_')}_meta_-{n_prev_meta}_dataset_ffill.csv")
                
#                 # # create for train test
#                 # cont_tbl_train = create_contigency_table(train,n_prev_meta, progress_bar=False, return_only_n_prev_meta=True)
#                 # cont_tbl_test = create_contigency_table(test,n_prev_meta, progress_bar=False, return_only_n_prev_meta=True)
                
#                 # cont_tbl_train.to_csv(path_data_dir + f"processed/train_contingency_table_ind_price_history_{asset.replace('.', '_')}_meta_-{n_prev_meta}_dataset_ffill.csv")
#                 # cont_tbl_test.to_csv(path_data_dir + f"processed/test_contingency_table_ind_price_history_{asset.replace('.', '_')}_meta_-{n_prev_meta}_dataset_ffill.csv")

# if __name__ == '__main__':
    
#     main()
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
from contigency_table import create_contigency_table

from joblib_progress import joblib_progress
from joblib import Parallel, delayed

# ### Definitions
path_data_dir = 'data/'

list_assets = ["PETR3.SA","PRIO3.SA", "VALE3.SA", "GGBR3.SA", "ABCB4.SA", "ITUB3.SA", "FLRY3.SA", "RADL3.SA"]

relevant_cols = ['Date', 'Close', 'Volume']

windows = [7,14,21]
max_seq_len = 70

seq_len_list = list(range(1, max_seq_len + 1))
# reverse list just for bigger n_prev_meta go first in the process
seq_len_list.reverse()

def process_n_prev_meta(n_prev_meta, asset, window):
    
    # data = pd.read_csv(path_data_dir + f"processed/price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
    train = pd.read_csv(path_data_dir + f"processed/train_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
    test = pd.read_csv(path_data_dir + f"processed/test_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
    
    ######################
    # dependent
    ######################
    column = f'meta_{window}'
    
    # Create for train test
    cont_tbl_train = create_contigency_table(train, n_prev_meta, column, progress_bar=False)
    cont_tbl_test = create_contigency_table(test, n_prev_meta, column, progress_bar=False)
    
    cont_tbl_train.to_csv(path_data_dir + f"processed/train_contingency_table_price_history_{asset.replace('.', '_')}_{column}_range({n_prev_meta})_dataset_ffill.csv")
    cont_tbl_test.to_csv(path_data_dir + f"processed/test_contingency_table_price_history_{asset.replace('.', '_')}_{column}_range({n_prev_meta})_dataset_ffill.csv")

def main():
    
    total_combinations = len(seq_len_list) * len(list_assets) * len(windows)
    
    with joblib_progress("Calculating square...", total=total_combinations):
        Parallel(n_jobs=-1)(delayed(process_n_prev_meta)(
            n_prev_meta, asset, window) 
                            for n_prev_meta in seq_len_list
                            for asset in list_assets
                            for window in windows
                        )
    
if __name__ == '__main__':
    main()
