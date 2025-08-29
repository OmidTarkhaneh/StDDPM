import torch
import numpy as np
import pandas as pd
from generate_dateframe import validate_and_fix_dates
import calendar
from tqdm import tqdm


def apply_date_constraints_to_sequences(generated_cat, cat_attrs, label_encoders):
    """
    Apply date validation constraints to generated categorical data
    """
    # Find indices of date-related features
    day_idx = cat_attrs.index('day') if 'day' in cat_attrs else None
    month_idx = cat_attrs.index('month') if 'month' in cat_attrs else None
    year_idx = cat_attrs.index('year') if 'year' in cat_attrs else None

    if day_idx is None or month_idx is None or year_idx is None:
        return generated_cat

    # Convert to numpy for easier manipulation
    generated_cat_np = generated_cat.cpu().numpy()

    # Process each sequence
    for seq_idx in range(generated_cat_np.shape[0]):
        for time_idx in range(generated_cat_np.shape[1]):
            # Get current encoded values
            day_encoded = generated_cat_np[seq_idx, time_idx, day_idx]
            month_encoded = generated_cat_np[seq_idx, time_idx, month_idx]
            year_encoded = generated_cat_np[seq_idx, time_idx, year_idx]

            # Decode to actual values
            try:
                day_actual = int(label_encoders['day'].inverse_transform([day_encoded])[0])
                month_actual = int(label_encoders['month'].inverse_transform([month_encoded])[0])
                year_actual = int(label_encoders['year'].inverse_transform([year_encoded])[0])

                # Validate and fix dates
                day_fixed, month_fixed, year_fixed = validate_and_fix_dates(day_actual, month_actual, year_actual)

                # Re-encode the fixed values
                day_fixed_encoded = label_encoders['day'].transform([str(day_fixed)])[0]
                month_fixed_encoded = label_encoders['month'].transform([str(month_fixed)])[0]
                year_fixed_encoded = label_encoders['year'].transform([str(year_fixed)])[0]

                # Update the generated data
                generated_cat_np[seq_idx, time_idx, day_idx] = day_fixed_encoded
                generated_cat_np[seq_idx, time_idx, month_idx] = month_fixed_encoded
                generated_cat_np[seq_idx, time_idx, year_idx] = year_fixed_encoded

            except (ValueError, IndexError):
                # If decoding fails, use safe defaults
                # Use middle values that are always valid
                safe_day = label_encoders['day'].transform(['15'])[0]  # 15th is always valid
                safe_month = label_encoders['month'].transform(['6'])[0]  # June
                safe_year = label_encoders['year'].transform(['2020'])[0]  # Safe year

                generated_cat_np[seq_idx, time_idx, day_idx] = safe_day
                generated_cat_np[seq_idx, time_idx, month_idx] = safe_month
                generated_cat_np[seq_idx, time_idx, year_idx] = safe_year

    return torch.tensor(generated_cat_np, dtype=torch.long, device=generated_cat.device)