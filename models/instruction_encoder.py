"""LSTM-based instruction encoder for BERT features"""

import torch
import torch.nn as nn


class InstructionEncoder(nn.Module):
    """Encodes pre-computed BERT instruction features"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # LSTM encoder
        self.encoder_rnn = nn.LSTM(
            input_size=config.embedding_size,  # 768 for BERT
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        
    def forward(self, observations):
        """
        Args:
            observations: dict with 'rxr_instruction' key containing BERT features
                         Shape: [batch_size, seq_len, 768]
        Returns:
            encoded features: [batch_size, seq_len, hidden_size]
        """
        instruction = observations["rxr_instruction"]
        
        # Get sequence lengths (non-zero entries)
        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1).cpu()
        
        # Pack sequence
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Encode
        output, (hidden, cell) = self.encoder_rnn(packed_seq)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        return output