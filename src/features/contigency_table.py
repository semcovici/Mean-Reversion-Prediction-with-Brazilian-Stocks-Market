import pandas as pd
from tqdm import tqdm

def process_row_contigency_table(row, data, n_prev_meta, return_only_n_prev_meta):
    
    meta = row['meta']
    date = row['Date']
    idx = row.index
    
    if idx < n_prev_meta:
        return None 
    
    
    new_row = {
        'Date': date,
        'Meta': meta
    }
    
    if return_only_n_prev_meta:
        
        new_row.update({f'Meta {- n_prev_meta}': data.loc[idx -(n_prev_meta),'meta']})
    else:
        new_row.update({f'Meta {-(meta + 1)}': data.loc[idx -(meta + 1),'meta'] for meta in range(n_prev_meta)})
        
    
    return pd.Series(new_row)



def create_contigency_table(
    data,
    n_prev_meta,
    col,
    progress_bar=True,
    as_probability=False,
    return_only_n_prev_meta = False
    
):
    # reset index (this code do not work without a sequential index)
    data.reset_index(drop = True, inplace=True)
    
    if return_only_n_prev_meta:
        table_schema = {'Date': [],'Meta':[], f'Meta {- n_prev_meta}': []}
    else:
        # create table
        table_schema = {'Date': [],'Meta':[]}
        table_schema.update({f'Meta {-(meta + 1)}': [] for meta in range(n_prev_meta)})
        
    df_prev_meta = pd.DataFrame(table_schema)
            
    for idx, row in tqdm(data.iterrows(), total = len(data), disable=not progress_bar):

        meta = row[col]
        date = row['Date']
        
        if idx < n_prev_meta:
            continue
        
        
        new_row = {
            'Date': date,
            'Meta': meta
        }
        
        if return_only_n_prev_meta:
            
            new_row.update({f'Meta {- n_prev_meta}': data.loc[idx -(n_prev_meta),col]})
        else:
            new_row.update({f'Meta {-(meta + 1)}': data.loc[idx -(meta + 1),col] for meta in range(n_prev_meta)})
            
        
        df_prev_meta.loc[len(df_prev_meta)] = new_row
        
        
    if return_only_n_prev_meta:
        cont_tbl = pd.crosstab(
        df_prev_meta['Meta'],
        [df_prev_meta[f'Meta {- n_prev_meta}']],
        margins=False
        )  
        
    else:
        cont_tbl = pd.crosstab(
        df_prev_meta['Meta'],
        [df_prev_meta[f'Meta {-(meta + 1)}'] for meta in range(n_prev_meta)],
        margins=False
        )
    
    if as_probability:        
        cont_tbl_probas = cont_tbl.apply(lambda x: [col/sum(x) for col in x])    
        return cont_tbl_probas
        
    
    return cont_tbl