import pandas as pd
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project src to the system path
import sys
sys.path.append('src/')
from model.evaluation import create_results_df
from model.config import create_experiment_configs_dummy

# ### Definitions
PROCESSED_DATA_PATH = 'data/processed/' 
PATH_REPORTS = 'reports/'
ASSETS = [
    "PETR3.SA", "PRIO3.SA", "VALE3.SA", 
    "GGBR3.SA", "ABCB4.SA", "ITUB3.SA", 
    "FLRY3.SA", "RADL3.SA"
]
MOVING_WINDOWS = [7, 14, 21]

def process_experiment(exp_name, config):
    """Process each experiment and generate the test results."""
    asset = config['asset']
    feature_col = config['feature_col']
    label_col = config['label_col']
    seq_len = config['seq_len']

    # Load test dataset
    test_dataset_path = f"{PROCESSED_DATA_PATH}test_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv"
    test_dataset = pd.read_csv(test_dataset_path, index_col=0)

    # Shift label column to create features
    for window in MOVING_WINDOWS:
        test_dataset[feature_col] = test_dataset[label_col].shift(1)

    # Remove the first rows based on sequence length
    test_dataset = test_dataset.iloc[seq_len:]

    # Prepare the results path
    path_results = f"{PATH_REPORTS}test_results/Dummy_model_{asset.replace('.', '_')}_features={feature_col}__label={label_col}_test_results.csv"

    # Generate predictions
    y_test = test_dataset[label_col]
    y_pred = test_dataset[feature_col].astype(int)

    # Create and save the results dataframe
    results_df = create_results_df(y_test, y_pred)
    print(f"Results saved to: {path_results}")
    results_df.to_csv(path_results, index=False)

def main():
    """Main function to execute all experiments."""
    experiment_configs = create_experiment_configs_dummy(ASSETS, MOVING_WINDOWS)

    for exp_name, config in tqdm(experiment_configs.items()):
        process_experiment(exp_name, config)

if __name__ == "__main__":
    main()