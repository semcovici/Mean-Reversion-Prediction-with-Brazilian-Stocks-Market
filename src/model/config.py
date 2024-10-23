from sklearn.preprocessing import StandardScaler

def create_experiment_configs_dummy(assets, windows):
    """Create a dictionary of experiment configurations."""
    experiment_configs = {}
    exp_id = 0

    for asset in assets:
        for window in windows:
            exp_id += 1
            experiment_configs[exp_id] = {
                "feature_col": f"past_diff_close_mean_z_score_{window}",
                "label_col": f"diff_close_mean_z_score_{window}",
                "window": window,
                "asset": asset,
                "seq_len": 1
            }

            exp_id += 1
            experiment_configs[exp_id] = {
                "feature_col": f"past_meta_{window}",
                "label_col": f'meta_{window}',
                "window": window,
                "asset": asset,
                "seq_len": 1
            }

    return experiment_configs


def create_experiment_configs_tf(assets, seq_len_list, moving_windows):
    """Cria um dicionário de configurações de experimentos."""
    experiment_configs = {}
    exp_id = 0

    for seq_len in seq_len_list:
        for asset in assets:
            for window in moving_windows: 
                for sub_conj_feats, label_col, prediction_type in [
                    ([f'meta_{window}'], f'meta_{window}', 'classification'),
                    ([f'diff_close_mean_z_score_{window}'], f'diff_close_mean_z_score_{window}', 'regression'),
                    ]:

                    for scaling_method in [
                        # StandardScaler(), 
                        None
                        ]:
                        for algorithm in ['LSTM_with_Attention', 'MLP', 'KAN']:
                            
                            exp_id += 1
                            
                            experiment_configs[exp_id] = {
                                "feature_cols": sub_conj_feats,
                                "label_col": label_col,
                                "seq_len": seq_len,
                                "asset": asset,
                                "scaling_method": scaling_method,
                                "algorithm": algorithm,
                                'prediction_type': prediction_type
                            }
                                
    return experiment_configs