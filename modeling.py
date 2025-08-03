"""
Train and evaluate models with PyCaret.
"""
import numpy as np
np.seterr(under='ignore')
import os
import pandas as pd
import joblib
from pycaret.classification import setup as setup_cls, create_model as create_cls, finalize_model as finalize_cls, pull as pull_cls
from pycaret.regression import setup as setup_reg, create_model as create_reg, finalize_model as finalize_reg, pull as pull_reg



def train_and_evaluate(df: pd.DataFrame, target: str, task: str, output_dir: str, test_df: pd.DataFrame=None):
    """
    Train XGB, LGBM, CatBoost models and save metrics & pickles.
    Returns dict of finalized models and metrics DataFrame.
    """
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    with np.errstate(all='ignore'):
        if task == 'regression':
            s = setup_reg(data=df, target=target, session_id=42, verbose=False)
            create_model = create_reg; finalize_model = finalize_reg; predict_model = None; pull = pull_reg
            metrics_file = os.path.join(output_dir, 'metrics_regression.xlsx')
        else:
            s = setup_cls(data=df, target=target, session_id=42, verbose=False)
            create_model = create_cls; finalize_model = finalize_cls; predict_model = None; pull = pull_cls
            metrics_file = os.path.join(output_dir, 'metrics_classification.xlsx')
        models = {}
        metrics_list = []
        for name in ['xgboost', 'lightgbm', 'catboost']:
            m = create_model(name)
            fm = finalize_model(m)
            joblib.dump(fm, os.path.join(output_dir, 'models', f'{name}.pkl'))
            met = pull().reset_index(drop=True)
            if 'Mean' in met.index:
                mean_row = met.loc['Mean']
            else:
                mean_row = met.mean(numeric_only=True)
            mean_row['model'] = name
            metrics_list.append(mean_row.to_frame().T)
            models[name] = fm
    metrics_df = pd.concat(metrics_list).reset_index(drop=True)
    metrics_df.to_excel(metrics_file, index=False)
    return models, metrics_df
