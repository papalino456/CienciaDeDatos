"""
Sentence pooling utilities for converting token embeddings to sentence embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over token embeddings with attention mask.
    
    Args:
        last_hidden_state: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
    
    Returns:
        pooled: [batch_size, hidden_size] - L2 normalized
    """
    # Expand mask to [batch_size, seq_len, hidden_size]
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Sum embeddings
    summed = torch.sum(last_hidden_state * mask, dim=1)
    
    # Count non-padding tokens
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    
    # Average
    pooled = summed / counts
    
    # L2 normalize
    pooled = F.normalize(pooled, p=2, dim=1)
    
    return pooled


def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Use [CLS] token embedding.
    
    Args:
        last_hidden_state: [batch_size, seq_len, hidden_size]
    
    Returns:
        pooled: [batch_size, hidden_size] - L2 normalized
    """
    cls_embedding = last_hidden_state[:, 0, :]
    return F.normalize(cls_embedding, p=2, dim=1)


def max_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Max pooling over token embeddings.
    
    Args:
        last_hidden_state: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
    
    Returns:
        pooled: [batch_size, hidden_size] - L2 normalized
    """
    # Set padding tokens to very negative value
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    last_hidden_state = last_hidden_state.clone()
    last_hidden_state[mask == 0] = -1e9
    
    # Max pool
    pooled = torch.max(last_hidden_state, dim=1)[0]
    
    # L2 normalize
    pooled = F.normalize(pooled, p=2, dim=1)
    
    return pooled


class SentenceEmbeddingModel(nn.Module):
    """Wrapper for BERT model with sentence pooling."""
    
    def __init__(self, bert_model, pooling_mode='mean'):
        super().__init__()
        self.bert = bert_model
        self.pooling_mode = pooling_mode
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass with pooling.
        
        Returns:
            sentence_embeddings: [batch_size, hidden_size]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        if self.pooling_mode == 'mean':
            return mean_pool(last_hidden_state, attention_mask)
        elif self.pooling_mode == 'cls':
            return cls_pool(last_hidden_state)
        elif self.pooling_mode == 'max':
            return max_pool(last_hidden_state, attention_mask)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
    
    def encode(self, input_ids, attention_mask):
        """Alias for forward for inference."""
        return self.forward(input_ids, attention_mask)

