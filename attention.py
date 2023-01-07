import torch
import torch.nn as nn

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        # Linear transformations for queries, keys, and values
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        # Linear transformation for final output
        self.output_linear = nn.Linear(d_model, d_model)
        
        # Number of attention heads
        self.n_heads = n_heads
        
        def forward(self, query, key, value, mask=None):
        # Split queries, keys, and values into n_heads dimensions
        query = self.query_linear(query).view(query.size(0), -1, self.n_heads, query.size(-1) // self.n_heads)
        key = self.key_linear(key).view(key.size(0), -1, self.n_heads, key.size(-1) // self.n_heads)
        value = self.value_linear(value).view(value.size(0), -1, self.n_heads, value.size(-1) // self.n_heads)
        
        # Transpose and dot-product attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        attention = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        attention = attention.softmax(dim=-1)
        output = torch.matmul(attention, value)
        
        # Concatenate and linear transform
        output = output.transpose(1, 2).contiguous().view(output.size(0), -1, self.n_heads * output.size(-1))
        output = self.output_linear(output)
        
        return output

