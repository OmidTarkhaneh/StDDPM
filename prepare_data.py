import pandas as pd
import numpy as np
import math
import random
from datetime import datetime, timedelta
import calendar
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def preprocess_data_czech(df):
    # Set seeds for reproducibility
    random.seed(0)
    np.random.seed(0)

    czech_date_parser = lambda x: datetime.strptime(str(x), "%y%m%d")
    df["datetime"] = df["date"].apply(czech_date_parser)

    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["dow"] = df["datetime"].dt.dayofweek
    df["year"] = df["datetime"].dt.year
    df["doy"] = df["datetime"].dt.dayofyear

    # Sort by account_id and datetime before calculating time differences
    df = df.sort_values(['account_id', 'datetime']).reset_index(drop=True)

    # Calculate time differences
    df["td"] = df.groupby("account_id")["datetime"].diff().dt.days.fillna(0.0)

    # Days till month end
    df["dtme"] = df.datetime.apply(lambda dt: calendar.monthrange(dt.year, dt.month)[1] - dt.day)

    # Raw amount calculation
    df['raw_amount'] = df.apply(lambda row: row['amount'] if row['type'] == 'CREDIT' else -row['amount'], axis=1)

    # Transaction code creation
    cat_code_fields = ['type', 'operation', 'k_symbol']
    TCODE_SEP = "__"
    tcode = df[cat_code_fields[0]].astype(str)
    for ccf in cat_code_fields[1:]:
        tcode += TCODE_SEP + df[ccf].astype(str)
    df["tcode"] = tcode

    # Day of month categories
    conditions = [
        (df['day'] >= 1) & (df['day'] <= 10),
        (df['day'] > 10) & (df['day'] <= 20),
        (df['day'] > 20) & (df['day'] <= 31)
    ]
    categories = ['first', 'middle', 'last']
    df['DoM_cat'] = np.select(conditions, categories, default='unknown')

    # Age group binning
    bin_edges = [17, 30, 40, 50, 60, 81]
    labels = ['18-30', '31-40', '41-50', '51-60', '61+']
    df['age_group'] = pd.cut(df['age'], bins=bin_edges, labels=labels, right=False)
    df['age_group'] = df['age_group'].astype('object')

    # Log amount scaling
    df['log_amount'] = np.log10(df['amount'] + 1)
    LOG_AMOUNT_SCALE = df['log_amount'].std()
    df['log_amount_sc'] = df['log_amount'] / LOG_AMOUNT_SCALE

    # Time difference scaling
    TD_SCALE = df['td'].std()
    df['td_sc'] = df['td'] / TD_SCALE

    return df







