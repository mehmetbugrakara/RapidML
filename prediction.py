"""
Generate predictions on test set and save to Excel.
"""
import os
import pandas as pd
from pycaret.classification import predict_model as predict_cls
from pycaret.regression import predict_model as predict_reg

def save_predictions(models: dict, test_df: pd.DataFrame, output_dir: str):
    """Run predict_model for each model and save combined results."""
    preds = []
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            dfp = predict_cls(model, data=test_df)
        else:
            dfp = predict_reg(model, data=test_df)
        dfp['model'] = name
        preds.append(dfp)
    result = pd.concat(preds).reset_index(drop=True)
    result.to_excel(os.path.join(output_dir, 'predictions.xlsx'), index=False)
