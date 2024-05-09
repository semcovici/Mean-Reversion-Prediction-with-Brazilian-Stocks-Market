from datetime import timedelta
import pandas as pd

def temporal_train_test_split(
    df,
    index_col = "Date",
    train_size = 0.8
):
    
    df.sort_values(index_col, inplace = True)
    df.reset_index(drop=True, inplace = True)
    
    idx_split = int(len(df) * train_size)

    #return train, test
    return df[:idx_split], df[idx_split:]

def split_data_by_year(
    data,
    time_col = "Date",
):
    
    data[time_col] = pd.to_datetime(data[time_col])
    
    data['year'] = data[time_col].dt.year
    
    # create dict with keys=year and values=data by year
    dataframes_by_year = {}
    for year, group in data.groupby('year'):
        
        dataframes_by_year[year] = group.drop(columns=['year'])
        
    return dataframes_by_year
    
    
    
    