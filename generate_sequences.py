import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy import stats
import warnings
from sequence_creation import apply_date_constraints_to_sequences

warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_sequences(model, diffuser, n_sequences, seq_len, n_cat_features, n_cat_tokens, n_num_features,
                      cat_attrs, label_encoders):
    model.eval()

    # Initialize categorical data randomly
    x_cat = torch.zeros(n_sequences, seq_len, n_cat_features, dtype=torch.long, device=device)
    for i, n_tokens in enumerate(n_cat_tokens):
        x_cat[:, :, i] = torch.randint(0, n_tokens, (n_sequences, seq_len), device=device)

    # Initialize numerical data with noise
    x_num = diffuser.sample_student_t((n_sequences, seq_len, n_num_features))

    with torch.no_grad():
        for t in tqdm(reversed(range(diffuser.total_steps)), desc="Generating"):
            timesteps = torch.full((n_sequences,), t, device=device, dtype=torch.long)

            # Get model predictions
            cat_outputs, num_output = model(x_cat, x_num, timesteps)

            # Denoise numerical data
            x_num = diffuser.sample(num_output, x_num, timesteps)

            # Update categorical data periodically
            if t % 100 == 0:
                for i, cat_out in enumerate(cat_outputs):
                    probs = torch.softmax(cat_out, dim=-1)
                    x_cat[:, :, i] = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(n_sequences, seq_len)

                # Apply date constraints after updating categorical data
                x_cat = apply_date_constraints_to_sequences(x_cat, cat_attrs, label_encoders)

    # Final date validation
    x_cat = apply_date_constraints_to_sequences(x_cat, cat_attrs, label_encoders)

    return x_cat, x_num