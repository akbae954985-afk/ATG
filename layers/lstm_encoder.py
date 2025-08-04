import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmSeq2SeqEncoder(nn.Module):
    """
    A PyTorch implementation of LSTM sequence-to-sequence encoder.
    This replaces the AllenNLP LstmSeq2SeqEncoder dependency.
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 batch_first=True, dropout=0.0, bidirectional=False):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
    def forward(self, inputs, mask=None):
        """
        Forward pass for the LSTM encoder.
        
        Args:
            inputs: Tensor of shape (batch_size, seq_len, input_size) if batch_first=True
            mask: Optional tensor of shape (batch_size, seq_len) indicating valid positions
            
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size * num_directions)
        """
        batch_size, seq_len, _ = inputs.size()
        
        if mask is not None:
            # Get sequence lengths from mask
            seq_lengths = mask.sum(dim=1).cpu()
            
            # Pack the sequence
            packed_inputs = pack_padded_sequence(
                inputs, seq_lengths, batch_first=self.batch_first, enforce_sorted=False
            )
            
            # Forward through LSTM
            packed_output, (hidden, cell) = self.lstm(packed_inputs)
            
            # Unpack the sequence
            output, _ = pad_packed_sequence(
                packed_output, batch_first=self.batch_first, total_length=seq_len
            )
        else:
            # No masking, standard forward pass
            output, (hidden, cell) = self.lstm(inputs)
            
        return output
        
    def get_output_dim(self):
        """Get the output dimension of the encoder."""
        return self.hidden_size * (2 if self.bidirectional else 1)