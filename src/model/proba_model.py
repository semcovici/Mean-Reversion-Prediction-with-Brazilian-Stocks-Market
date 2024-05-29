def get_past_meta(
    df,
    row,
    n_past_meta,
    name_meta_col = "meta"
):
    idx = row.name
        
    list_past_meta = df.loc[idx-n_past_meta:idx - 1, name_meta_col].to_list()
    
    list_past_meta.reverse()
    
    if len(list_past_meta) == 1:
        return str(list_past_meta[0])
        
    return tuple([str(i) for i in list_past_meta])


def predict(
    past_meta,
    df_probas
):
    
    # if this past_meta is not in df_proba
    if past_meta not in df_probas.columns:
        return float('nan')
    
    
    # create probability dict
    dict_probas = df_probas.loc[:,past_meta].to_dict()
    
    
    # get the meta with greater proba
    pred_meta = max(dict_probas, key=dict_probas.get)
    
    return pred_meta