import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EndpointSpanExtractor(nn.Module):
    """
    A span extractor that extracts spans by their endpoints.
    This replaces the AllenNLP EndpointSpanExtractor.
    """
    
    def __init__(self, input_dim, combination="x,y", num_width_embeddings=None, 
                 span_width_embedding_dim=None, bucket_widths=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.combination = combination
        self.num_width_embeddings = num_width_embeddings
        self.span_width_embedding_dim = span_width_embedding_dim
        self.bucket_widths = bucket_widths
        
        # Parse combination string
        self._combination_parts = combination.split(",")
        
        # Calculate output dimension
        output_dim = 0
        for part in self._combination_parts:
            if part in ["x", "y"]:
                output_dim += input_dim
            elif part == "x*y":
                output_dim += input_dim
            elif part == "x+y":
                output_dim += input_dim
            elif part == "x-y":
                output_dim += input_dim
                
        if span_width_embedding_dim and num_width_embeddings:
            self.span_width_embedding = nn.Embedding(
                num_width_embeddings, span_width_embedding_dim
            )
            output_dim += span_width_embedding_dim
        else:
            self.span_width_embedding = None
            
        self._output_dim = output_dim
        
    def forward(self, sequence_tensor, span_indices, sequence_mask=None, span_indices_mask=None):
        """
        Extract spans from the sequence tensor.
        
        Args:
            sequence_tensor: (batch_size, seq_len, input_dim)
            span_indices: (batch_size, num_spans, 2) - start and end indices
            sequence_mask: Optional mask for sequence
            span_indices_mask: Optional mask for spans
            
        Returns:
            (batch_size, num_spans, output_dim)
        """
        batch_size, num_spans, _ = span_indices.size()
        
        # Extract start and end positions
        span_starts = span_indices[:, :, 0]  # (batch_size, num_spans)
        span_ends = span_indices[:, :, 1]    # (batch_size, num_spans)
        
        # Get start and end representations
        # Expand indices for gathering
        batch_indices = torch.arange(batch_size, device=span_indices.device).unsqueeze(1).expand(-1, num_spans)
        
        start_embeddings = sequence_tensor[batch_indices, span_starts]  # (batch_size, num_spans, input_dim)
        end_embeddings = sequence_tensor[batch_indices, span_ends]      # (batch_size, num_spans, input_dim)
        
        # Combine according to combination string
        combined_tensors = []
        
        for part in self._combination_parts:
            if part == "x":
                combined_tensors.append(start_embeddings)
            elif part == "y":
                combined_tensors.append(end_embeddings)
            elif part == "x*y":
                combined_tensors.append(start_embeddings * end_embeddings)
            elif part == "x+y":
                combined_tensors.append(start_embeddings + end_embeddings)
            elif part == "x-y":
                combined_tensors.append(start_embeddings - end_embeddings)
                
        # Add width embeddings if configured
        if self.span_width_embedding is not None:
            span_widths = span_ends - span_starts
            if self.bucket_widths:
                span_widths = self._bucket_widths(span_widths)
            span_widths = torch.clamp(span_widths, 0, self.num_width_embeddings - 1)
            width_embeddings = self.span_width_embedding(span_widths)
            combined_tensors.append(width_embeddings)
            
        return torch.cat(combined_tensors, dim=-1)
        
    def get_output_dim(self):
        return self._output_dim
        
    def _bucket_widths(self, widths):
        """Bucket the widths into discrete buckets."""
        # Simple bucketing strategy
        return torch.clamp(widths, 0, self.num_width_embeddings - 1)


class SelfAttentiveSpanExtractor(nn.Module):
    """
    A span extractor that uses self-attention over the span.
    This replaces the AllenNLP SelfAttentiveSpanExtractor.
    """
    
    def __init__(self, input_dim, num_width_embeddings=None, span_width_embedding_dim=None,
                 bucket_widths=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_width_embeddings = num_width_embeddings
        self.span_width_embedding_dim = span_width_embedding_dim
        self.bucket_widths = bucket_widths
        
        # Attention parameters
        self.global_attention = nn.Linear(input_dim, 1)
        
        output_dim = input_dim
        if span_width_embedding_dim and num_width_embeddings:
            self.span_width_embedding = nn.Embedding(
                num_width_embeddings, span_width_embedding_dim
            )
            output_dim += span_width_embedding_dim
        else:
            self.span_width_embedding = None
            
        self._output_dim = output_dim
        
    def forward(self, sequence_tensor, span_indices, sequence_mask=None, span_indices_mask=None):
        """
        Extract spans using self-attention.
        
        Args:
            sequence_tensor: (batch_size, seq_len, input_dim)
            span_indices: (batch_size, num_spans, 2) - start and end indices
            
        Returns:
            (batch_size, num_spans, output_dim)
        """
        batch_size, num_spans, _ = span_indices.size()
        
        span_embeddings = []
        
        for i in range(batch_size):
            batch_span_embeddings = []
            for j in range(num_spans):
                start_idx = span_indices[i, j, 0].item()
                end_idx = span_indices[i, j, 1].item()
                
                if start_idx >= 0 and end_idx >= start_idx:
                    # Extract span tokens
                    span_tokens = sequence_tensor[i, start_idx:end_idx+1, :]  # (span_len, input_dim)
                    
                    if span_tokens.size(0) > 0:
                        # Compute attention weights
                        attention_weights = self.global_attention(span_tokens)  # (span_len, 1)
                        attention_weights = F.softmax(attention_weights, dim=0)
                        
                        # Weighted sum
                        span_embedding = torch.sum(attention_weights * span_tokens, dim=0)  # (input_dim,)
                    else:
                        span_embedding = torch.zeros(self.input_dim, device=sequence_tensor.device)
                else:
                    span_embedding = torch.zeros(self.input_dim, device=sequence_tensor.device)
                    
                batch_span_embeddings.append(span_embedding)
                
            span_embeddings.append(torch.stack(batch_span_embeddings))
            
        span_embeddings = torch.stack(span_embeddings)  # (batch_size, num_spans, input_dim)
        
        # Add width embeddings if configured
        if self.span_width_embedding is not None:
            span_starts = span_indices[:, :, 0]
            span_ends = span_indices[:, :, 1]
            span_widths = span_ends - span_starts
            if self.bucket_widths:
                span_widths = self._bucket_widths(span_widths)
            span_widths = torch.clamp(span_widths, 0, self.num_width_embeddings - 1)
            width_embeddings = self.span_width_embedding(span_widths)
            span_embeddings = torch.cat([span_embeddings, width_embeddings], dim=-1)
            
        return span_embeddings
        
    def get_output_dim(self):
        return self._output_dim
        
    def _bucket_widths(self, widths):
        """Bucket the widths into discrete buckets."""
        return torch.clamp(widths, 0, self.num_width_embeddings - 1)


class BidirectionalEndpointSpanExtractor(nn.Module):
    """
    A span extractor that extracts bidirectional endpoint representations.
    This replaces the AllenNLP BidirectionalEndpointSpanExtractor.
    """
    
    def __init__(self, input_dim, forward_combination="x,y", backward_combination="x,y"):
        super().__init__()
        
        self.input_dim = input_dim
        self.forward_combination = forward_combination
        self.backward_combination = backward_combination
        
        # Assume input comes from bidirectional LSTM, so split into forward and backward
        assert input_dim % 2 == 0, "Input dimension must be even for bidirectional"
        self.forward_dim = input_dim // 2
        self.backward_dim = input_dim // 2
        
        # Create endpoint extractors for forward and backward
        self.forward_extractor = EndpointSpanExtractor(
            self.forward_dim, combination=forward_combination
        )
        self.backward_extractor = EndpointSpanExtractor(
            self.backward_dim, combination=backward_combination
        )
        
        self._output_dim = self.forward_extractor.get_output_dim() + self.backward_extractor.get_output_dim()
        
    def forward(self, sequence_tensor, span_indices, sequence_mask=None, span_indices_mask=None):
        """
        Extract bidirectional endpoint spans.
        
        Args:
            sequence_tensor: (batch_size, seq_len, input_dim)
            span_indices: (batch_size, num_spans, 2) - start and end indices
            
        Returns:
            (batch_size, num_spans, output_dim)
        """
        # Split into forward and backward representations
        forward_sequence = sequence_tensor[:, :, :self.forward_dim]
        backward_sequence = sequence_tensor[:, :, self.forward_dim:]
        
        # Extract spans from both directions
        forward_spans = self.forward_extractor(forward_sequence, span_indices, sequence_mask, span_indices_mask)
        backward_spans = self.backward_extractor(backward_sequence, span_indices, sequence_mask, span_indices_mask)
        
        # Concatenate
        return torch.cat([forward_spans, backward_spans], dim=-1)
        
    def get_output_dim(self):
        return self._output_dim