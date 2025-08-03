"""
Data loading utilities.
"""
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load CSV or Excel file into DataFrame."""
    if path.lower().endswith('.csv'):
        return pd.read_csv(path)
    else:
        return pd.read_excel(path)
