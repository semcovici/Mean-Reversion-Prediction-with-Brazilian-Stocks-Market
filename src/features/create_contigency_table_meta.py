
# ### Imports


import plotly.graph_objects as go

import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt


sns.set_style('darkgrid')

# ### Definitions


path_data_dir = 'data/'

list_assets = ["PETR3.SA","PRIO3.SA", "VALE3.SA", "GGBR3.SA", "ABCB4.SA", "ITUB3.SA", "FLRY3.SA"]

relevant_cols = ['Date', 'Close', 'Volume']


n_prev_meta = 10

for asset in list_assets:
        
    data = pd.read_csv(path_data_dir + f'processed/price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv')

    data.reset_index(drop = True, inplace=True)

    table_schema = {'Date': [],'Meta':[]}
    table_schema.update({f'Meta {-(meta + 1)}': [] for meta in range(n_prev_meta)})

    df_prev_meta = pd.DataFrame(table_schema)

    for idx, row in tqdm(data.iterrows(), total = len(data), desc= f'{asset}'):

        meta = row['meta']
        date = row['Date']
        
        if idx < n_prev_meta:
            continue
        
        new_row = {
            'Date': date,
            'Meta': meta
        }
        
        new_row.update({f'Meta {-(meta + 1)}': data.loc[idx -(meta + 1),'meta'] for meta in range(n_prev_meta)})
        
        df_prev_meta.loc[len(df_prev_meta)] = new_row
        
    cont_tbl = pd.crosstab(
    df_prev_meta['Meta'],
    [df_prev_meta[f'Meta {-(meta + 1)}'] for meta in range(n_prev_meta)],
    margins=False
    )
    
    cont_tbl.to_csv(path_data_dir + f"processed/contingency_table_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")


