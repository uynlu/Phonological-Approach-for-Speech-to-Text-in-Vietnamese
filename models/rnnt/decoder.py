import torch.nn as nn
import torch


# Language model
class Decoder(nn.Module):
    def __init__(self, vocab, hidden_size, num_layers, dropout_prob):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.embedding_size = len(self.vocab) - 4
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.start_symbol = self.vocab.bos_idx

        self.embedding = nn.Embedding(
            num_embeddings=len(self.vocab),
            embedding_dim=self.embedding_size,
            padding_idx=self.vocab.pad_idx
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            dropout=self.dropout_prob,
            bidirectional=True
        )
    
    def forward(self, inputs):  # inputs.size(): (batch_size, text_len)
        batch_size = inputs.size()[0]
        text_len = inputs.size()[1]
        hidden_state = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=inputs.device)
        cell = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=inputs.device)
        outputs = []
        for i in range(text_len):
            if i == 0:
                decoder_input = (torch.tensor([self.start_symbol] * batch_size, device=inputs.device)).unsqueeze(-1)  # decoder_input.size(): (batch_size, 1)
            else:
                decoder_input = inputs[:, i].unsqueeze(-1)  # decoder_input.size(): (batch_size, 1)
            output, hidden_state, cell = self.forward_one_step(decoder_input, hidden_state, cell)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)  # outputs.size(): (batch_size, text_len + 1, hidden_size * 2)
        outputs = outputs.squeeze(2)

        return outputs


    def forward_one_step(
            self,
            input,  # input.size(): (batch_size, 1)
            previous_hidden_state,  # hidden_state.size(): num_layers * 2, batch_size, hidden_size
            previos_cell
        ):
        embeddeding = (self.dropout(self.embedding(input)))  # embeddeding.size(): (batch_size, 1, embedding_size)
        output, (hidden_state, cell) = self.lstm(embeddeding, (previous_hidden_state, previos_cell))
        # output.size(): (batch_size, 1, 2 * hidden_size)
        # hidden_state.size(): (2 * num_layers, batch_size, hidden_size)

        return output, hidden_state, cell
