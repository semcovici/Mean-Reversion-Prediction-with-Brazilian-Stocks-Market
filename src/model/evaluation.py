
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, classification_report

import pandas as pd
def get_classification_report(y_test, y_pred):
    '''Source: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format'''
    report = classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report.reset_index(drop=False).rename({'index': 'class'}, axis =1 )

def regression_metrics(y_test, y_pred):
    metrics = {
        'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_pred),
        'Mean Squared Error (MSE)': mean_squared_error(y_test, y_pred),
        'Root Mean Squared Error (RMSE)': mean_squared_error(y_test, y_pred, squared=False),
        'R-squared (R2)': r2_score(y_test, y_pred),
        'Mean Absolute Percentage Error (MAPE)': mean_absolute_percentage_error(y_test, y_pred)
    }
    
    metrics_df = pd.DataFrame(metrics, index=[0])
    return metrics_df


def create_results_df(y_test, y_pred):
    
    results_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred
    })
    
    return results_df
    