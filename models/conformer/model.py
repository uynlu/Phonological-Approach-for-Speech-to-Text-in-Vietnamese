import torch
from torch import nn

from builders.model_builder import META_MODEL
from .encoder import ConformerEncoder
from .decoder import LSTMDecoder

def generate_padding_mask(sequences: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    '''
        sequences: (bs, seq_len, dim)
    '''
    
    if len(sequences.shape) == 2: # (bs, seq_len)
        __seq = sequences.unsqueeze(dim=-1) # (bs, seq_len, 1)
    else:
        __seq = sequences
    
    mask = (torch.sum(__seq, dim=-1) == (padding_value * __seq.shape[-1])).to(torch.bool) # (b_s, seq_len)
    
    return mask # (bs, seq_len)

@META_MODEL.register()
class ConFormer(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.pad_idx = vocab.pad_idx
        self.d_model = config.d_model
        
        encoder_config = config.encoder
        self.encoder = ConformerEncoder(
               d_input=encoder_config.d_input,
               d_model=encoder_config.d_model,
               num_layers=encoder_config.num_layers,
               conv_kernel_size=encoder_config.conv_kernel_size,
               feed_forward_residual_factor=encoder_config.ffwd_residual_factor,
               feed_forward_expansion_factor=encoder_config.ffwd_expansion_factor,
               num_heads=encoder_config.num_heads,
               dropout=encoder_config.dropout
		)

        decoder_config = config.decoder
        self.decoder = LSTMDecoder(
            d_encoder=decoder_config.d_model,
            d_decoder=decoder_config.d_model,
            num_classes=vocab.size,
            num_layers=decoder_config.num_layers
        )

        self.loss_fn = nn.CTCLoss(blank=self.pad_idx, zero_infinity=True)

    def forward(self, voice_tensor: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        voice_tensor = voice_tensor.transpose(-1, -2)
        logits = self.forward_step(voice_tensor)

        input_lengths = logits.shape[:-1]
        logits = logits.permute((1, 0, -1)) # (len, bs, vocab_size)
        target_lengths = (labels == self.pad_idx).sum(dim=-1)

        loss = self.loss_fn(logits, labels, input_lengths, target_lengths)

        return logits, loss
    
    def forward_step(self, voice_tensor: torch.Tensor) -> torch.Tensor:
        mask = generate_padding_mask(voice_tensor, padding_value=self.pad_idx)
        features = self.encoder(voice_tensor, mask)
        logits = self.decoder(features)

        return logits
    
    def generate(self, voice_tensor: torch.Tensor):
        logits = self.forward_step(voice_tensor)
        predicted_ids = logits.argmax(dim=-1)

        return predicted_ids
