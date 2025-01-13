import torch.nn as nn
import torch.nn.functional as F


class Joiner(nn.Module):
    MODES = {
        "multiplicative": lambda outputs_encoder, outputs_decoder: outputs_encoder * outputs_decoder,
        "additive": lambda outputs_encoder, outputs_decoder: outputs_encoder + outputs_decoder,
        # "cat": lambda outputs_encoder, outputs_decoder: torch.cat((outputs_encoder, outputs_decoder), dim=1)
    }

    def __init__(self, hidden_size, vocab, mode):
        super(Joiner, self).__init__()

        self.vocab = vocab
        self.join = self.MODES[mode]

        self.linear = nn.Linear(
            in_features=hidden_size * 2,
            out_features=len(self.vocab)
        )

    def forward(
            self,
            outputs_encoder,  # outputs_encoder.size(): (batch_size, num_samples, hidden_size * 2)
            outputs_decoder,  # outputs_decoder.size(): (batch_size, text_len + 1, hidden_size * 2)
        ):
        outputs_encoder = outputs_encoder.unsqueeze(2)  # outputs_encoder.size(): (batch_size, num_samples, 1, hidden_size * 2)
        outputs_decoder = outputs_decoder.unsqueeze(1)  # outputs_decoder.size(): (batch_size, 1, text_len + 1, hidden_size * 2)

        outputs = self.join(outputs_encoder, outputs_decoder)  # outputs.size(): (batch_size, num_samples, text_len + 1, hidden_size * 2)

        outputs = self.linear(outputs)  # outputs.size(): (batch_size, num_samples, text_len + 1, len_vocab)
        
        outputs = F.softmax(outputs, dim=-1)

        return outputs
