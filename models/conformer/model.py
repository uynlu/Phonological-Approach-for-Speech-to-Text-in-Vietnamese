import torch
from torch import nn

from builders.model_builder import META_MODEL
from .encoder import ConformerEncoder
from .decoder import LSTMDecoder
from utils.instance import InstanceList

def generate_3D_padding_mask(sequences: torch.Tensor, lengths: list, padding_value: int = 0) -> torch.Tensor:
    mask = torch.ones(sequences.shape[0], sequences.shape[1], sequences.shape[1]) # (bs, len, len)
    for i, l in enumerate(lengths):
        mask[i, :, :l] = padding_value

    return mask.bool()

def generate_2D_padding_mask(sequences: torch.Tensor, lengths: list, padding_value: int = 0) -> torch.Tensor:
    mask = torch.ones(sequences.shape[0], sequences.shape[1]) # (bs, len)
    for i, l in enumerate(lengths):
        mask[i, :l] = padding_value

    return mask.bool()

def generate_casual_mask(seq_len: int) -> torch.Tensor:
    input_tensor = torch.zeros((seq_len, seq_len))
    mask = torch.triu(input_tensor, diagonal=1).bool()

    return mask

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
        labels = items.shifted_right_labels
        input_lengths = items.input_length
        input_lengths = [((length - 1) // 2 - 1) // 2 for length in input_lengths] # account for subsampling of time dimension
        target_lengths = items.target_length

        logits = self.forward_step(voice_tensor, input_lengths)
        logits = logits.permute((1, 0, -1)) # (len, bs, vocab_size)

        loss = self.loss_fn(logits, labels, input_lengths, target_lengths)

        return logits, loss
    
    def forward_step(self, voice_tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = generate_3D_padding_mask(voice_tensor, lengths).to(self.device)
        features = self.encoder(voice_tensor, mask)
        logits = self.decoder(features)
        print(logits)
        return logits
    
    def generate(self, items: InstanceList):
        voice_tensor = items.voice
        input_lengths = items.input_length
        input_lengths = [((length - 1) // 2 - 1) // 2 for length in input_lengths] # account for subsampling of time dimension

        logits = self.forward_step(voice_tensor, input_lengths)
        predicted_ids = logits.argmax(dim=-1)

        return predicted_ids

@META_MODEL.register()
class ConFormer_seq2seq(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.pad_idx = vocab.pad_idx
        self.d_model = config.d_model
        self.device = config.device
        self.vocab_size = vocab.size
        
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
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=decoder_config.embedding_dim,
            padding_idx=self.pad_idx
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=decoder_config.d_model,
                nhead=decoder_config.nhead,
                dim_feedforward=decoder_config.d_ffwd,
                dropout=decoder_config.dropout,
                batch_first=True
            ),
            num_layers=decoder_config.num_layers
        )
        self.fc = nn.Linear(
            in_features=decoder_config.d_model,
            out_features=self.vocab_size
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, items: InstanceList) -> tuple[torch.Tensor, torch.Tensor]:
        voice_tensor = items.voice
        labels = items.labels
        shifted_right_labels = items.shifted_right_labels
        input_lengths = items.input_length
        input_lengths = [((length - 1) // 2 - 1) // 2 for length in input_lengths] # account for subsampling of time dimension

        encoder_features = self.forward_encoder(voice_tensor, input_lengths)
        logits = self.forward_decoder(encoder_features, input_lengths, shifted_right_labels, input_lengths)
        logits = logits.permute((1, 0, -1)) # (len, bs, vocab_size)
        logits = self.fc(logits)

        loss = self.loss_fn(logits.reshape(-1, self.vocab_size), labels.reshape(-1, ))

        return logits, loss
    
    def forward_encoder(self, voice_tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = generate_3D_padding_mask(voice_tensor, lengths).to(self.device)
        features = self.encoder(voice_tensor, mask)

        return features
    
    def forward_decoder(self, encoder_features: torch.Tensor, encoder_lengths: list, input_ids: torch.LongTensor, input_lengths: list) -> torch.Tensor:
        input_padding_mask = generate_2D_padding_mask(input_ids, input_lengths, padding_value=self.pad_idx).to(self.device)
        input_casual_mask = generate_casual_mask(input_ids.shape[-1]).to(self.device)
        voice_mask = generate_2D_padding_mask(encoder_features, encoder_lengths).to(self.device)

        input_embs = self.embedding(input_ids.long())

        logits: torch.Tensor = self.decoder(
            tgt=input_embs,
            memory=encoder_features,
            tgt_mask=input_casual_mask,
            memory_key_padding_mask=voice_mask,
            tgt_key_padding_mask=input_padding_mask
        )

        return logits
    
    def generate(self, items: InstanceList):
        voice_tensor = items.voice
        shifted_right_labels = items.shifted_right_labels
        input_lengths = items.input_length
        input_lengths = [((length - 1) // 2 - 1) // 2 for length in input_lengths] # account for subsampling of time dimension

        encoder_features = self.forward_encoder(voice_tensor, input_lengths)
        logits = self.forward_decoder(encoder_features, input_lengths, shifted_right_labels, input_lengths)
        logits = self.fc(logits)
        predicted_ids = logits.argmax(dim=-1)

        return predicted_ids
