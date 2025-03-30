import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding for Audio Spectrogram
    """
    def __init__(
        self, 
        img_size=(128, 1024),  # (freq bins, time frames)
        patch_size=(16, 16), 
        in_channels=1, 
        embed_dim=768
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Args:
            x: Shape (B, C, F, T)
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, F, T = x.shape
        
        # Ensure input size matches expected size
        assert F <= self.img_size[0] and T <= self.img_size[1], \
            f"Input size ({F}, {T}) doesn't match model expected size {self.img_size}"
        
        # If input is smaller than img_size, pad to img_size
        if F < self.img_size[0] or T < self.img_size[1]:
            padding = (0, self.img_size[1] - T, 0, self.img_size[0] - F)
            x = F.pad(x, padding)
        
        # Project patches
        x = self.proj(x)  # (B, embed_dim, F//patch_size, T//patch_size)
        
        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x


class Attention(nn.Module):
    """
    Multi-head Attention module
    """
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # For storing attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Calculate query, key, value vectors
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, C//num_heads)
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Store attention weights for visualization
        self.attention_weights = attn.detach()
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Project to output
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    MLP module for Transformer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block
    """
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4., 
        qkv_bias=True, 
        drop=0., 
        attn_drop=0., 
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AudioSpectrogramTransformer(nn.Module):
    """
    Audio Spectrogram Transformer model
    """
    def __init__(
        self, 
        img_size=(128, 1024),  # (freq bins, time frames)
        patch_size=(16, 16), 
        in_channels=1,
        num_classes=5,
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=True,
        representation_size=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        pos_drop_rate=0.
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate
            ) for _ in range(depth)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Representation layer (pre-logits)
        if representation_size:
            self.representation = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.representation = nn.Identity()
        
        # Head
        self.head = nn.Linear(self.representation.out_features, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize patch embedding
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize MLP weights in transformer blocks
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward_features(self, x, return_attention=False):
        """
        Extract features using the transformer
        """
        B = x.shape[0]
        
        # Convert spectrogram to patches
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed[:, :(x.size(1))]
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        attention_maps = []
        for block in self.blocks:
            x = block(x)
            if return_attention and hasattr(block.attn, 'attention_weights'):
                attention_maps.append(block.attn.attention_weights)
        
        # Apply normalization
        x = self.norm(x)
        
        # Use class token as representation
        x = x[:, 0]
        
        # Apply representation layer
        x = self.representation(x)
        
        if return_attention:
            return x, attention_maps
        
        return x
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        """
        if return_attention:
            x, attention_maps = self.forward_features(x, return_attention)
            x = self.head(x)
            return x, attention_maps
        
        x = self.forward_features(x)
        x = self.head(x)
        return x


# For testing
if __name__ == "__main__":
    # Create a test input
    batch_size = 4
    input_channels = 1
    input_freq = 128
    input_time = 1024
    
    x = torch.randn(batch_size, input_channels, input_freq, input_time)
    
    # Create model with smaller dimensions for testing
    model = AudioSpectrogramTransformer(
        img_size=(input_freq, input_time),
        patch_size=(16, 16),
        in_channels=input_channels,
        num_classes=5,
        embed_dim=192,
        depth=4,
        num_heads=3
    )
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with attention maps
    output, attention_maps = model(x, return_attention=True)
    
    print(f"Number of attention maps: {len(attention_maps)}")
    print(f"Attention map shape: {attention_maps[0].shape}") 