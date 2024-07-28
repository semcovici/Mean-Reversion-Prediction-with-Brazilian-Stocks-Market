
from sklearn.metrics import classification_report
import pandas as pd
def get_classification_report(y_test, y_pred):
    '''Source: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format'''
    report = classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report


def create_results_df(y_test, y_pred):
    
    results_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred
    })
    
    return results_df
    