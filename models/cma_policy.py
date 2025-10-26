"""Cross-Modal Attention (CMA) Policy - Model only, no training code"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space

from .instruction_encoder import InstructionEncoder
from .resnet_encoders import VlnResnetDepthEncoder, TorchVisionResNet50


class CMAPolicy:
    """CMA Policy wrapper"""
    
    def __init__(self, observation_space, action_space, model_config):
        self.net = CMANet(observation_space, model_config, action_space.n)
        self.action_space = action_space
        
    def act(self, observations, rnn_states, prev_actions, masks, deterministic=False):
        features, rnn_states = self.net(observations, rnn_states, prev_actions, masks)
        
        # Simple linear layer for action distribution
        logits = self.net.action_head(features)
        
        if deterministic:
            action = logits.argmax(dim=-1, keepdim=True)
        else:
            action = torch.distributions.Categorical(logits=logits).sample().unsqueeze(-1)
        
        return action, rnn_states
    
    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
    
    def to(self, device):
        self.net.to(device)
        return self
    
    def eval(self):
        self.net.eval()
        return self


class CMANet(nn.Module):
    """Cross-Modal Attention Network"""
    
    def __init__(self, observation_space, model_config, num_actions):
        super().__init__()
        self.model_config = model_config
        self.num_recurrent_layers = model_config.num_recurrent_layers
        
        # Instruction encoder
        self.instruction_encoder = InstructionEncoder(model_config)
        
        # Visual encoders
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.depth_encoder_output_size,
            backbone=model_config.depth_encoder_backbone,
        )
        
        self.rgb_encoder = TorchVisionResNet50(
            output_size=model_config.rgb_encoder_output_size,
            backbone=model_config.rgb_encoder_backbone,
        )
        
        # Previous action embedding
        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
        
        # State encoder (RNN)
        hidden_size = model_config.hidden_size
        rnn_input_size = (
            model_config.rgb_encoder_output_size +
            model_config.depth_encoder_output_size +
            32  # prev action
        )
        
        self.state_encoder = nn.LSTM(
            rnn_input_size,
            hidden_size,
            num_layers=model_config.num_recurrent_layers,
            batch_first=True,
        )
        
        # Attention layers
        self.text_attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.visual_attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
        # Action head
        self.action_head = nn.Linear(hidden_size, num_actions)
        
    def forward(self, observations, rnn_states, prev_actions, masks):
        # Encode instruction
        instruction_embedding = self.instruction_encoder(observations)
        
        # Encode visuals
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)
        
        # Previous action
        prev_action_emb = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )
        
        # Concatenate features
        x = torch.cat([rgb_embedding, depth_embedding, prev_action_emb], dim=1)
        
        # RNN
        x = x.unsqueeze(1)  # Add sequence dimension
        x, rnn_states_out = self.state_encoder(x, self._unpack_rnn_state(rnn_states))
        x = x.squeeze(1)
        
        # Cross-modal attention (simplified)
        x = x + instruction_embedding.mean(dim=1)  # Simple fusion
        
        return x, self._pack_rnn_state(rnn_states_out)
    
    def _unpack_rnn_state(self, rnn_states):
        """Convert flat tensor to LSTM state tuple"""
        batch_size = rnn_states.size(0)
        num_layers = self.num_recurrent_layers
        hidden_size = rnn_states.size(2)
        
        h = rnn_states[:, :num_layers, :]
        c = rnn_states[:, num_layers:, :]
        
        return (h.transpose(0, 1).contiguous(), c.transpose(0, 1).contiguous())
    
    def _pack_rnn_state(self, rnn_states):
        """Convert LSTM state tuple to flat tensor"""
        h, c = rnn_states
        return torch.cat([h.transpose(0, 1), c.transpose(0, 1)], dim=1)