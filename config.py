"""Simple configuration for RxR evaluation"""

class Config:
    def __init__(self):
        # Model config
        self.hidden_size = 512
        self.rnn_type = "LSTM"
        self.num_recurrent_layers = 2
        
        # Instruction encoder
        self.instruction_encoder_type = "LSTM"
        self.embedding_size = 768  # BERT embedding size
        self.bidirectional = True
        
        # Visual encoders
        self.rgb_encoder_output_size = 256
        self.depth_encoder_output_size = 128
        self.rgb_encoder_backbone = "resnet50"
        self.depth_encoder_backbone = "resnet50"
        
        # Environment
        self.image_width = 640
        self.image_height = 480
        self.num_actions = 6  # STOP, FORWARD, LEFT, RIGHT, LOOK_UP, LOOK_DOWN
        
        # Evaluation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 1
        
    def to_dict(self):
        return self.__dict__

import torch