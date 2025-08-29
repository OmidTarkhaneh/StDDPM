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
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, diffuser, dataloader, epochs, optimizer, scheduler):
    model.train()
    cat_criterion = nn.CrossEntropyLoss()
    num_criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_cat, batch_num in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            batch_cat, batch_num = batch_cat.to(device), batch_num.to(device)

            # Sample random timesteps
            timesteps = diffuser.sample_random_timesteps(batch_cat.shape[0])

            # Add noise to numerical data
            noisy_num, noise_target = diffuser.add_t_noise(batch_num, timesteps)

            # Forward pass
            cat_outputs, num_output = model(batch_cat, noisy_num, timesteps)

            # Calculate losses
            cat_loss = sum(cat_criterion(cat_out.view(-1, cat_out.size(-1)),
                                       batch_cat[:, :, i].view(-1))
                          for i, cat_out in enumerate(cat_outputs))
            num_loss = num_criterion(num_output, noise_target)
            total_loss = cat_loss + num_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        scheduler.step()