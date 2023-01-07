import torch

from transformer import Transformer
from data_utils import create_dataloader

def evaluate_bleu(model, dataloader, src_field, tgt_field, device):
    model.eval()
    bleu_score = 0.
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = create_mask(src, tgt)
            output = model.predict(src, src_mask, tgt_mask, max_length=tgt.size(1))
            output_str = [tgt_field.vocab.itos[i] for i in output]
            tgt_str = [tgt_field.vocab.itos[i] for i in tgt[:, 1:].squeeze(0)]
            bleu_score += bleu(output_str, tgt_str)
            
    return bleu_score / len(dataloader)

def bleu(output_str, tgt_str):
    # Calculate BLEU score
    # ...
    
    return score

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = create_mask(src, tgt)
            output = model(src, tgt, src_mask, tgt_mask)
            loss = criterion(output, tgt[:, 1:])
            total_loss += loss.item()
            
    return total_loss / len(dataloader)
