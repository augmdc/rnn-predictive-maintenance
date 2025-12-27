import torch.nn as nn

class SimpleRNN(nn.Module):
    """
    Simple RNN model for time series prediction.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units in RNN
        output_size: Number of output features
        num_layers: Number of RNN layers (default: 1)
        rnn_type: Type of RNN - 'RNN', 'LSTM', or 'GRU' (default: 'LSTM')
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        num_layers: int = 1,
        rnn_type: str = 'LSTM',
        dropout: float = 0.0
    ):
        super(SimpleRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Create RNN layer based on type
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(
                input_size, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Invalid rnn_type: {rnn_type}. Choose 'RNN', 'LSTM', or 'GRU'.")
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Pass through RNN layer
        # rnn_out shape: (batch_size, sequence_length, hidden_size)
        rnn_out, _ = self.rnn(x)
        
        # Take the output from the last time step
        # last_output shape: (batch_size, hidden_size)
        last_output = rnn_out[:, -1, :]
        
        # Pass through fully connected layer
        # output shape: (batch_size, output_size)
        output = self.fc(last_output)
        
        return output
