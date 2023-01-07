import torch
import torch.optim as optim

from transformer import Transformer
from data_utils import create_dataloader

def train(model, train_dataloader, val_dataloader, optimizer, criterion, n_epochs, device):
    model.train()
    for epoch in range(n_epochs):
        for src, tgt in train_dataloader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = create_mask(src, tgt)

            optimizer.zero_grad()
            output = model(src, tgt, src_mask, tgt_mask)
            loss = criterion(output, tgt[:, 1:])
            loss.backward()
            optimizer.step()
            
        # Validate model on validation set
        val_loss = evaluate(model, val_dataloader, criterion, device)
        print(f'Epoch {epoch+1}/{n_epochs}: Validation loss = {val_loss:.3f}')
        

