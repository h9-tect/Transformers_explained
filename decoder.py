import torch
import torch.nn as nn

from attention import MultiHeadedAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        
        # Multi-headed attention layers
        self.self_attention = MultiHeadedAttention(d_model, n_heads)
        self.encoder_attention = MultiHeadedAttention(d_model, n_heads)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        def forward(self, inputs, encoding, src_mask, tgt_mask):
        # Self-attention
        output = self.self_attention(inputs, inputs, inputs, tgt_mask)
        
        # Apply dropout and layer normalization
        output = self.dropout(output)
        output = self.layer_norm1(output + inputs)
        
        # Encoder-decoder attention
        output = self.encoder_attention(output, encoding, encoding, src_mask)
        
        # Apply dropout and layer normalization
        output = self.dropout(output)
        output = self.layer_norm2(output + inputs)
        
        # Feedforward network
        output = self.ffn(output)
        
        # Apply dropout and layer normalization
        output = self.dropout(output)
        output = self.layer_norm3(output + inputs)
        
        return output
    
class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dropout):
        super().__init__()
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
    def forward(self, inputs, encoding, src_mask, tgt_mask):
        # Pass input through decoder layers
        for layer in self.layers:
            inputs = layer(inputs, encoding, src_mask, tgt_mask)
            
        return inputs

