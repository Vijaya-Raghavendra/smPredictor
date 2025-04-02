from torch import nn

class LstmModule(nn.Module):
    """
    A class that handles the LSTM model for stock market prediction
    """
    def __init__(self, n_features, hidden_size=1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=False,
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)
        return self.fc(output[-1, :])