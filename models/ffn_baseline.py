"""
Baseline PyTorch implementation of GEGLU FFN.

GEGLU FFN formula:
    u = Wu @ x          # [B, 4096] -> [B, 12288]
    v = Wv @ x          # [B, 4096] -> [B, 12288]
    h = GELU(u) ⊙ v     # element-wise multiplication
    y = Wo @ h          # [B, 12288] -> [B, 4096]

Where:
    - Wu: [4096, 12288] weight matrix
    - Wv: [4096, 12288] weight matrix
    - Wo: [12288, 4096] weight matrix
    - ⊙: element-wise multiplication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GEGLU_FFN(nn.Module):
    """
    Gated Linear Unit with GELU activation for Feed-Forward Network.
    
    Args:
        hidden_size (int): Input and output dimension. Default: 4096
        intermediate_size (int): Intermediate dimension. Default: 12288
    """
    
    def __init__(self, hidden_size=4096, intermediate_size=12288):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Three linear layers without bias
        self.Wu = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.Wv = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.Wo = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x):
        """
        Forward pass of GEGLU FFN.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, hidden_size]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, hidden_size]
        """
        # Step 1: Linear transformations to intermediate size
        u = self.Wu(x)      # (B, 12288)
        v = self.Wv(x)      # (B, 12288)
        
        # Step 2: Apply GELU activation to u
        g = F.gelu(u)       # (B, 12288)
        
        # Step 3: Element-wise multiplication (gating)
        h = g * v           # (B, 12288)
        
        # Step 4: Project back to hidden size
        y = self.Wo(h)      # (B, 4096)
        
        return y
    
    def get_weights(self):
        """
        Get the weight matrices for testing and verification.
        
        Returns:
            dict: Dictionary containing Wu, Wv, Wo weight matrices
        """
        return {
            'Wu': self.Wu.weight.data,
            'Wv': self.Wv.weight.data,
            'Wo': self.Wo.weight.data
        }
