import numpy as np
import pandas as pd
import calendar
from datetime import datetime, timedelta
import calendar
from tqdm import tqdm

def validate_and_fix_dates(day, month, year):
    """
    Validate and fix invalid date combinations
    Returns corrected day, month, year
    """
    # Ensure values are within valid ranges
    year = max(1995, min(2025, year))  # Reasonable year range
    month = max(1, min(12, month))

    # Get the maximum valid day for the given month and year
    max_day = calendar.monthrange(year, month)[1]
    day = max(1, min(max_day, day))

    return day, month, year

def create_dataframe_with_date_constraints(generated_cat, generated_num, cat_attrs, num_attrs,
                                         label_encoders, num_scaler):
    """Create dataframe from generated sequences with date validation"""
    final_sequences = []

    for seq_idx in range(generated_cat.shape[0]):
        seq_num = num_scaler.inverse_transform(generated_num[seq_idx].cpu().numpy())
        seq_cat = generated_cat[seq_idx].cpu().numpy()

        seq_df = pd.DataFrame()

        # Add numerical features
        for i, col in enumerate(num_attrs):
            seq_df[col] = seq_num[:, i]

        # Add categorical features with additional validation
        for i, col in enumerate(cat_attrs):
            try:
                decoded_values = label_encoders[col].inverse_transform(seq_cat[:, i])
                seq_df[col] = decoded_values
            except ValueError:
                # Handle any remaining encoding issues
                print(f"Warning: Issues with decoding {col}, using safe defaults")
                if col == 'day':
                    seq_df[col] = [15] * len(seq_cat)  # Safe default day
                elif col == 'month':
                    seq_df[col] = [6] * len(seq_cat)   # Safe default month
                elif col == 'year':
                    seq_df[col] = [2020] * len(seq_cat) # Safe default year
                else:
                    # For other categorical features, use the most common value
                    most_common = label_encoders[col].classes_[0]
                    seq_df[col] = [most_common] * len(seq_cat)

        # Additional date validation at DataFrame level
        if all(col in seq_df.columns for col in ['day', 'month', 'year']):
            for idx in range(len(seq_df)):
                day_val = int(seq_df.loc[idx, 'day'])
                month_val = int(seq_df.loc[idx, 'month'])
                year_val = int(seq_df.loc[idx, 'year'])

                day_fixed, month_fixed, year_fixed = validate_and_fix_dates(day_val, month_val, year_val)

                seq_df.loc[idx, 'day'] = day_fixed
                seq_df.loc[idx, 'month'] = month_fixed
                seq_df.loc[idx, 'year'] = year_fixed

        seq_df['account_id'] = seq_idx + 1000  # Start from 1000 to avoid conflicts
        final_sequences.append(seq_df)

    return pd.concat(final_sequences, ignore_index=True)