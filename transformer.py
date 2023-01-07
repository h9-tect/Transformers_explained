import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, dropout):
        super().__init__()
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Encoder and decoder
        self.encoder = Encoder(d_model, n_layers, n_heads, dropout)
        self.decoder = Decoder(d_model, n_layers, n_heads, dropout)
        
        # Linear layer to produce final output
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        # Embed input sequences
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        
        # Pass through encoder and decoder
        encoding = self.encoder(src, src_mask)
        output = self.decoder(tgt, encoding, src_mask, tgt_mask)
        
        # Produce final output
        output = self.output_linear(output)
        
        return output
