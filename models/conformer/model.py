import torch
from torch import nn

from builders.model_builder import META_MODEL
from .encoder import ConformerEncoder
from .decoder import LSTMDecoder
from utils.instance import InstanceList

def generate_padding_mask(sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(sequences.shape[0], sequences.shape[1], sequences.shape[1]) # (bs, len, len)
    for i, l in enumerate(lengths):
        mask[i, :, :l] = 0

    return mask.bool()

@META_MODEL.register()
class ConFormer(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.pad_idx = vocab.pad_idx
        self.d_model = config.d_model
        self.device = config.device
        
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
            d_encoder=encoder_config.d_model,
            d_decoder=decoder_config.d_model,
            num_classes=vocab.size,
            num_layers=decoder_config.num_layers
        )

        self.loss_fn = nn.CTCLoss(blank=self.pad_idx, zero_infinity=True)

    def forward(self, items: InstanceList) -> tuple[torch.Tensor, torch.Tensor]:
        voice_tensor = items.voice
        labels = items.labels
        input_lengths = items.input_length
        input_lengths = [((length - 1) // 2 - 1) // 2 for length in input_lengths] # account for subsampling of time dimension
        target_lengths = items.target_length

        logits = self.forward_step(voice_tensor, input_lengths)
        logits = logits.permute((1, 0, -1)) # (len, bs, vocab_size)

        loss = self.loss_fn(logits, labels, input_lengths, target_lengths)

        return logits, loss
    
    def forward_step(self, voice_tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = generate_padding_mask(voice_tensor, lengths).to(self.device)
        features = self.encoder(voice_tensor, mask)
        logits = self.decoder(features)

        return logits
    
    def generate(self, items: InstanceList):
        voice_tensor = items.voice
        input_lengths = items.input_length
        input_lengths = [((length - 1) // 2 - 1) // 2 for length in input_lengths] # account for subsampling of time dimension

        logits = self.forward_step(voice_tensor, input_lengths)
        predicted_ids = logits.argmax(dim=-1)

        return predicted_ids
