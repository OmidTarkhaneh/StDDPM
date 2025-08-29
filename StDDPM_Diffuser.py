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
import math

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def piecewise_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02, warmup_frac=0.3):
    warmup_steps = int(num_timesteps * warmup_frac)
    betas1 = torch.linspace(beta_start, beta_end * 0.5, warmup_steps)
    betas2 = torch.linspace(beta_end * 0.5, beta_end, num_timesteps - warmup_steps)
    return torch.cat([betas1, betas2])


class StudentTDDPMDiffuser(object):
    def __init__(self, total_steps=1000, beta_start=1e-4, beta_end=0.02, device='cpu', df=10):
        self.total_steps = total_steps
        self.device = device
        self.df = df

        scale = 1000 / total_steps
        beta_start = scale * beta_start
        beta_end = scale * beta_end
        betas=piecewise_schedule(total_steps)
        self.alphas = (1.0 - betas).to(device)
        self.betas = betas.to(device)
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

    def sample_random_timesteps(self, n: int):
        return torch.randint(low=1, high=self.total_steps, size=(n,), device=self.device)

    def sample_student_t(self, shape):
        x = torch.randn(shape, device=self.device)
        df_sample = max(3.0, float(self.df))
        gamma_shape = df_sample / 2.0
        gamma_samples = torch.tensor(stats.gamma.rvs(gamma_shape, scale=2.0, size=shape[0]),
                                   dtype=torch.float32, device=self.device).view(-1, 1, 1)
        scaling = torch.sqrt(torch.tensor(df_sample / (df_sample - 2.0), device=self.device))
        return scaling * x / torch.sqrt(gamma_samples / df_sample)

    def add_t_noise(self, x_num, t):
        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alphas_hat[t])[:, None, None]
        noise_num = self.sample_student_t(x_num.shape)
        return sqrt_alpha_hat * x_num + sqrt_one_minus_alpha_hat * noise_num, noise_num

    def sample(self, model_out, z_norm, timesteps):
        sqrt_alpha_t = torch.sqrt(self.alphas[timesteps])[:, None, None]
        betas_t = self.betas[timesteps][:, None, None]
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - self.alphas_hat[timesteps])[:, None, None]
        epsilon_t = torch.sqrt(self.betas[timesteps][:, None, None])

        random_noise = self.sample_student_t(z_norm.shape)
        random_noise[timesteps == 0] = 0.0

        model_mean = ((1 / sqrt_alpha_t) * (z_norm - (betas_t * model_out / sqrt_one_minus_alpha_hat_t)))
        return model_mean + (epsilon_t * random_noise)


class TemporalLSTMSynthesizer(nn.Module):
    def __init__(self, n_cat_features, n_cat_tokens, cat_emb_dim, n_num_features, hidden_dim=128):
        super().__init__()
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(n_cat_tokens[i], cat_emb_dim) for i in range(n_cat_features)
        ])

        total_input_dim = n_cat_features * cat_emb_dim + n_num_features

        self.time_embed = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.input_projection = nn.Linear(total_input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True, dropout=0.1)

        self.cat_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_cat_tokens[i]) for i in range(n_cat_features)
        ])
        self.num_head = nn.Linear(hidden_dim, n_num_features)

    def embed_time(self, timesteps, dim_out=64):
        half = dim_out // 2
        freqs = torch.exp(-math.log(1000) * torch.arange(0, half, dtype=torch.float32) / half).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x_cat, x_num, timesteps):
        batch_size, seq_len, _ = x_num.shape

        # Embed categorical features
        cat_emb = torch.cat([self.cat_embeddings[i](x_cat[:, :, i])
                           for i in range(self.n_cat_features)], dim=-1)

        # Combine features
        x = torch.cat([cat_emb, x_num], dim=-1)
        x_proj = self.input_projection(x)

        # Add time embedding
        time_emb = self.time_embed(self.embed_time(timesteps))
        x_with_time = x_proj + time_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # LSTM processing
        lstm_out, _ = self.lstm(x_with_time)

        # Generate outputs
        cat_outputs = [head(lstm_out) for head in self.cat_heads]
        num_output = self.num_head(lstm_out)

        return cat_outputs, num_output




class TemporalSequentialDataset(Dataset):
    def __init__(self, df, cat_attrs, num_attrs, sequence_length=30, min_seq_length=20):
        self.sequences = []
        for account_id in tqdm(df['account_id'].unique(), desc="Creating sequences"):
            account_data = df[df['account_id'] == account_id].sort_values('datetime').reset_index(drop=True)
            if len(account_data) >= min_seq_length:
                for start_idx in range(0, len(account_data) - sequence_length + 1, sequence_length//2):
                    end_idx = min(start_idx + sequence_length, len(account_data))
                    seq_data = account_data.iloc[start_idx:end_idx]

                    # Pad sequences if they're shorter than sequence_length
                    if len(seq_data) < sequence_length:
                        # Pad with the last row repeated
                        padding_needed = sequence_length - len(seq_data)
                        last_row = seq_data.iloc[-1:].copy()
                        padding = pd.concat([last_row] * padding_needed, ignore_index=True)
                        seq_data = pd.concat([seq_data, padding], ignore_index=True)

                    self.sequences.append({
                        'cat_data': seq_data[cat_attrs].values,
                        'num_data': seq_data[num_attrs].values
                    })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.LongTensor(seq['cat_data']), torch.FloatTensor(seq['num_data'])