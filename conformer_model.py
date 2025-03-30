import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class ConformerBlock(nn.Module):
    """
    Conformer block consists of:
    1. Feed Forward Module
    2. Multi-Head Self-Attention Module 
    3. Convolution Module
    4. Feed Forward Module
    With residual connections and layer normalization
    """
    def __init__(self, dim, num_heads, kernel_size, dropout=0.1, expansion_factor=4):
        super().__init__()
        
        # Feed Forward Module 1
        self.ff1 = FeedForward(dim, expansion_factor, dropout)
        self.ff1_norm = nn.LayerNorm(dim)
        
        # Multi-Head Self-Attention Module
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        
        # Convolution Module
        self.conv = ConformerConvModule(dim, kernel_size, dropout)
        self.conv_norm = nn.LayerNorm(dim)
        
        # Feed Forward Module 2
        self.ff2 = FeedForward(dim, expansion_factor, dropout)
        self.ff2_norm = nn.LayerNorm(dim)
        
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        # First Feed Forward Module with 0.5 scaling
        x = x + 0.5 * self.ff1(self.ff1_norm(x))
        
        # Multi-Head Self-Attention Module
        x = x + self.attn(self.attn_norm(x), mask=mask)
        
        # Convolution Module
        x = x + self.conv(self.conv_norm(x))
        
        # Second Feed Forward Module with 0.5 scaling
        x = x + 0.5 * self.ff2(self.ff2_norm(x))
        
        # Final layer normalization
        return self.final_norm(x)


class FeedForward(nn.Module):
    """
    Feed Forward module for Conformer
    """
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module with relative positional encoding
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        b, n, c = x.shape  # batch, sequence length, channels
        
        # Get query, key, value vectors
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # Split heads
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Project to output
        return self.to_out(out)


class ConformerConvModule(nn.Module):
    """
    Convolution module for Conformer
    """
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        
        # Pointwise conv
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        
        # 1D depthwise conv
        self.depthwise_conv = nn.Conv1d(
            dim, 
            dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2,
            groups=dim
        )
        
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        
        # Pointwise conv
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Convert [batch, seq_len, channels] to [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # Pointwise conv 1
        x = self.pointwise_conv1(x)
        
        # GLU activation
        x = F.glu(x, dim=1)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        
        # BatchNorm + activation
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise conv 2
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Convert back to [batch, seq_len, channels]
        return x.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
    """
    def __init__(self, dim, max_len=5000):
        super().__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Conformer(nn.Module):
    """
    Conformer model for heart sound classification
    """
    def __init__(
        self, 
        input_dim=128,  # mel frequency bins
        hidden_dim=256, 
        num_classes=5,  # 5 classes as per dataset
        num_layers=6, 
        num_heads=4, 
        kernel_size=31, 
        dropout=0.1,
        expansion_factor=4
    ):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                hidden_dim, 
                num_heads, 
                kernel_size, 
                dropout,
                expansion_factor
            ) for _ in range(num_layers)
        ])
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, num_classes)
        
        # For attention visualization
        self.attention_weights = None
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: Input spectrogram [batch, channels, freq, time]
            mask: Optional mask for padding
            return_attention: Whether to return attention weights for visualization
        """
        # Convert input shape [batch, channels, freq, time] -> [batch, time, freq]
        b, c, f, t = x.shape
        x = x.transpose(2, 3).reshape(b, t, f)
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_enc(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply Conformer blocks
        attentions = []
        for layer in self.layers:
            x = layer(x, mask)
            if return_attention:
                # For capturing attention weights (would need modification in actual implementation)
                attentions.append(x)
        
        # Global average pooling across time dimension
        x = x.transpose(1, 2)  # [batch, hidden_dim, time]
        x = self.pool(x).squeeze(-1)  # [batch, hidden_dim]
        
        # Output layer
        logits = self.output(x)
        
        if return_attention:
            return logits, attentions
        
        return logits


# For testing
if __name__ == "__main__":
    # Create a test input
    batch_size = 4
    input_channels = 1
    input_freq = 128
    input_time = 400
    
    x = torch.randn(batch_size, input_channels, input_freq, input_time)
    
    # Create model
    model = Conformer(input_dim=input_freq)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 