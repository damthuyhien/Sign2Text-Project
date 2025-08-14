import torch
import torch.nn as nn

class SignSequenceNet(nn.Module):
    def __init__(self, num_classes=29):
        super(SignSequenceNet, self).__init__()
        # CNN extract features
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # LSTM sequence model
        self.lstm = nn.LSTM(input_size=32*7*7, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)                   # [B, 32, 7, 7]
        x = x.view(batch_size, 1, -1)     # [B, seq_len=1, features]
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])      # output cuối LSTM
        return out
