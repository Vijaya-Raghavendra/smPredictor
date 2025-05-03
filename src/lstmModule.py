from torch import nn
import torch.nn.functional as F

class LstmModule(nn.Module):

    """
    A class that handles the LSTM model for stock market prediction
    """
        
    def __init__(self, n_features, hidden_size=32):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        return

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)
        x = F.relu(self.fc1(output[-1, :]))
        return self.fc2(x)
