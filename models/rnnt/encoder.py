import torch.nn as nn


# Acoustic model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            dropout=self.dropout_prob,
            bidirectional=True
        )

    def forward(self, inputs):  # inputs.size(): (batch_size, num_samples, num_mels)
        outputs, _ = self.lstm(inputs)  # outputs.size(): (batch_size, num_samples, hidden_size * 2)
        return outputs
