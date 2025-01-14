from torch import nn

class LSTMDecoder(nn.Module):
    '''
    LSTM Decoder

    Parameters:
        d_encoder (int): Output dimension of the encoder
        d_decoder (int): Hidden dimension of the decoder
        num_layers (int): Number of LSTM layers to use in the decoder
        num_classes (int): Number of output classes to predict

    Inputs:
        x (Tensor): (batch_size, time, d_encoder)

    Outputs:
        Tensor (batch_size, time, num_classes): Class prediction logits

    '''
    def __init__(self, d_encoder, d_decoder, num_layers, num_classes):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=d_encoder, hidden_size=d_decoder, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(d_decoder, num_classes)

    def forward(self, inputs):
        outputs_lstm, _ = self.lstm(inputs)
        logits = self.linear(outputs_lstm)
        outputs = nn.functional.log_softmax(logits, dim=-1)
        return outputs
    