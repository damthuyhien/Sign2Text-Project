import torch
import torch.nn as nn

class SignSequenceNet(nn.Module):
    def __init__(self, input_size=21*3, hidden_size=128, num_classes=26):
        super(SignSequenceNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, keypoints]
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # lấy output cuối cùng của sequence
        return out
