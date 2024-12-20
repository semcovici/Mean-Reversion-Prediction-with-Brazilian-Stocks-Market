import yfinance as yahooFinance


list_ativos =["PETR3.SA","PRIO3.SA", "VALE3.SA", "GGBR3.SA", "ABCB4.SA", "ITUB3.SA", "FLRY3.SA", "RADL3.SA"]
start_dt="2010-11-24" 
end_dt="2024-03-01"

path_price_history = 'data/raw/price_history_{ativo}.xlsx'


if __name__ == '__main__':
    
    
    # get data from yahoo finance
    data = yahooFinance.download(
        list_ativos,
        start = start_dt,
        end = end_dt,
        group_by="ticker"
        )

    data.dropna(inplace=True)
    
    # save files 
    for ativo in list_ativos:
        
        price_hist =  data[ativo]
        
        ativo = ativo.replace('.', '_')
                
        price_hist.to_excel(path_price_history.format(ativo = ativo))
        