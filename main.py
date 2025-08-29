import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

import pandas as pd
import numpy as np
import math
import random
import os
from datetime import datetime, timedelta
import calendar
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from generate_dateframe import create_dataframe_with_date_constraints
from prepare_data import preprocess_data_czech
from train  import train_model
from generate_sequences import generate_sequences
from StDDPM_Diffuser import StudentTDDPMDiffuser, TemporalLSTMSynthesizer, TemporalSequentialDataset
from Evalaution import compute_ngram_metrics, comapre_unidist_cat,comapre_unidist_cont, compute_2d_categorical_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
sequence_length = 80
min_seq_length = 20
cat_emb_dim = 8
mlp_layers = [128, 128]
diffusion_steps = 1000  # Reduced for efficiency
epochs = 50  # Reduced for efficiency
batch_size = 550
learning_rate = 2e-4
n_sequences = 5000


# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Loading data...")
    try:
        real = pd.read_csv('/Users/omidtarkhaneh/Desktop/StDDPM/StDDPM_source/tr_by_acct_w_age.csv')
    except FileNotFoundError:
        print("Error: CSV file not found.")
        exit()

    # Preprocess data
    raw_data = preprocess_data_czech(real)
    cat_attrs = ['tcode', 'dow', 'month', 'day', 'year', 'DoM_cat']
    num_attrs = ['amount', 'raw_amount', 'td']

    df_processed = raw_data[cat_attrs + num_attrs + ['account_id', 'datetime']].copy()

    # Encode categorical features
    label_encoders = {}
    n_cat_tokens = []
    for attr in cat_attrs:
        le = LabelEncoder()
        df_processed[attr] = le.fit_transform(df_processed[attr].astype(str))
        label_encoders[attr] = le
        n_cat_tokens.append(len(le.classes_))

    # Scale numerical features
    num_scaler = QuantileTransformer(output_distribution='normal', random_state=seed)
    df_processed[num_attrs] = num_scaler.fit_transform(df_processed[num_attrs])

    # Create dataset and dataloader
    dataset = TemporalSequentialDataset(df_processed, cat_attrs, num_attrs, sequence_length, min_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TemporalLSTMSynthesizer(
        n_cat_features=len(cat_attrs),
        n_cat_tokens=n_cat_tokens,
        cat_emb_dim=cat_emb_dim,
        n_num_features=len(num_attrs)
    ).to(device)

    # Initialize diffuser and optimizer
    diffuser = StudentTDDPMDiffuser(total_steps=diffusion_steps, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters...")

    # Train the model
    train_model(model, diffuser, dataloader, epochs, optimizer, scheduler)

    # Generate sequences
    print("Generating sequences...")
    generated_cat, generated_num = generate_sequences(
        model, diffuser, n_sequences, sequence_length,
        len(cat_attrs), n_cat_tokens, len(num_attrs),
        cat_attrs, label_encoders
    )

    # Create final dataframe
    synth_data = create_dataframe_with_date_constraints(
        generated_cat, generated_num, cat_attrs, num_attrs,
        label_encoders, num_scaler
    )

    # Prepare data for evaluation
    synth_sorted = synth_data[['account_id', 'raw_amount', 'amount', 'td', 'tcode', 'year', 'day', 'month']].copy()




# Evaluation

synth_sorted['datetime'] = pd.to_datetime(synth_sorted[['year', 'month', 'day']])
synth=synth_sorted
# synth['datetime'] = pd.to_datetime(synth['datetime'], errors='coerce')
synth['datetime'] = pd.to_datetime(synth['datetime'])
synth['datetime'] = pd.to_datetime(synth['datetime']).dt.date
synth=synth.dropna()
czech_date_parser = lambda x: datetime.strptime(str(x), "%Y-%m-%d")
synth["datetime"] = synth["datetime"].apply(czech_date_parser)
synth["month"] = synth["datetime"].dt.month
synth["day"] = synth["datetime"].dt.day
synth["dow"] =  synth["datetime"].dt.dayofweek
synth["year"] = synth["datetime"].dt.year

synth["td"] = synth[["account_id", "datetime"]].groupby("account_id").diff()
synth["td"] = synth["td"].apply(lambda x: x.days)
synth["td"].fillna(0.0, inplace=True)

# synth.rename(columns={'days_passed': 'td'}, inplace=True)
synth['type'] = synth['tcode'].str.split('__').str[0]
synth['raw_amount'] = synth.apply(lambda row: row['amount'] if row['type'] == 'CREDIT' else -row['amount'], axis=1)
synth["dtme"] = synth.datetime.apply(lambda dt: calendar.monthrange(dt.year, dt.month)[1] - dt.day)
synth_sorted = synth.sort_values(['account_id', 'year', 'month', 'day'])
synth_cf = synth[["account_id", "month", "raw_amount", "year"]].groupby(["account_id", "month", "year"],as_index=False)["raw_amount"].sum()


df2 = raw_data[['account_id','tcode', 'datetime','amount', 'raw_amount']]
real = df2.copy()
real["month"] = real["datetime"].dt.month
real["day"] = real["datetime"].dt.day
real["dow"] =  real["datetime"].dt.dayofweek
real["year"] = real["datetime"].dt.year

real["td"] = real[["account_id", "datetime"]].groupby("account_id").diff()
real["td"] = real["td"].apply(lambda x: x.days)
real["td"].fillna(0.0, inplace=True)
real['type'] = real['tcode'].str.split('__').str[0]

# dtme - days till month end
real["dtme"] = real.datetime.apply(lambda dt: calendar.monthrange(dt.year, dt.month)[1] - dt.day)

real_cf = real[["account_id", "month", "raw_amount", "year"]].groupby(["account_id", "month", "year"],as_index=False)["raw_amount"].sum()
real_sorted = real.sort_values(['account_id', 'year', 'month', 'day'])

# Results

##########################################
combo_df, result = compute_ngram_metrics(real_sorted, synth_sorted, 'tcode', 3)
print(result)


##########################################

CAT_FIELDS = ['tcode', 'day', 'month']
result_jst_cat = {}
for field in CAT_FIELDS:
    result_jst_cat[field] = comapre_unidist_cat(real, synth_sorted, field)

print(result_jst_cat)


##########################################

CONT_FIELDS = ["amount", "td"]
CF_FIELD = 'raw_amount'
#compare univariate distribution of continuous columns
comapre_unidist_cont(CONT_FIELDS,CF_FIELD, real, synth_sorted, real_cf, synth_cf)

##########################################
field1='tcode'
field2='day'

compute_2d_categorical_metrics(real_sorted, synth_sorted, field1, field2)