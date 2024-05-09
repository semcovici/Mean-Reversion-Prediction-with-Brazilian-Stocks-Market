import pandas as pd
from tqdm import tqdm
def create_contigency_table(
    data,
    n_prev_meta,
    progress_bar=True,
    as_probability=False
):

    data.reset_index(drop = True, inplace=True)

    table_schema = {'Date': [],'Meta':[]}
    table_schema.update({f'Meta {-(meta + 1)}': [] for meta in range(n_prev_meta)})

    df_prev_meta = pd.DataFrame(table_schema)

    for idx, row in tqdm(data.iterrows(), total = len(data), disable=not progress_bar):

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
    
    if as_probability:        
        cont_tbl_probas = cont_tbl.apply(lambda x: [col/sum(x) for col in x])    
        return cont_tbl_probas
        
    
    return cont_tbl