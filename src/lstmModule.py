from torch import nn

class LstmModule(nn.Module):

    """
    A class that handles the LSTM model for stock market prediction
    """

    # def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
    #     super().__init__()
    #     self.lstm = nn.LSTM(
    #         input_size=n_features,
    #         hidden_size=hidden_size,
    #         num_layers=num_layers,
    #         dropout=dropout,
    #         batch_first=True
    #     )
    #     self.fc = nn.Linear(hidden_size, 1)
    #     self.bn = nn.BatchNorm1d(hidden_size)
    #     self.activation = nn.LeakyReLU()
    #     return
    

    # def forward(self, X):
    #     if X.dim() == 2:
    #         # Assuming shape is (seq_len, n_features), add batch dimension
    #         X = X.unsqueeze(0)
    #     elif X.dim() == 1:
    #         # Assuming flat vector, reshape to (1, seq_len=1, features)
    #         X = X.view(1, 1, -1)

    #     output, _ = self.lstm(X)
    #     last_step = output[:, -1, :]
    #     y_pred = self.fc(self.activation(self.bn(last_step)))
    #     return y_pred.squeeze(-1)


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
            return
    
    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)
        return self.fc(output[-1, :])