import torch
import torch.nn as nn

from attention import MultiHeadedAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        
        # Multi-headed attention layer
        self.attention = MultiHeadedAttention(d_model, n_heads)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, mask):
        # Self-attention
        output = self.attention(inputs, inputs, inputs, mask)
        
        # Apply dropout and layer normalization
        output = self.dropout(output)
        output = self.layer_norm1(inputs + output)
        
        # Feedforward network
        output = self.ffn(output)
        
        # Apply dropout and layer normalization
        output = self.dropout(output)
        output = self.layer_norm2(output + inputs)
        
        return output
    
class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dropout):
        super().__init__()
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
    def forward(self, inputs, mask):
        # Pass input through encoder layers
        for layer in self.layers:
            inputs = layer(inputs, mask)
            
        return inputs
