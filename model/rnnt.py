import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.text.vocabulary import Vocabulary
import numpy as np
from torchaudio.functional import rnnt_loss


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


# Language model
class Decoder(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, num_layers, dropout_prob):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.start_symbol = self.vocab.get_index("<sos>")  # self.start_symbol.shape: (1, 3) ([[41, 163, 9]])

        self.embedding_consonant = nn.Embedding(
            num_embeddings=self.vocab.len()[0],
            embedding_dim=self.embedding_size,
            padding_idx=self.vocab.consonant_2_index.get("<pad>")
        )
        self.embedding_vowel = nn.Embedding(
            num_embeddings=self.vocab.len()[1],
            embedding_dim=self.embedding_size
        )
        self.embedding_tone = nn.Embedding(
            num_embeddings=self.vocab.len()[2],
            embedding_dim=self.embedding_size
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
    
    def forward(self, inputs):  # inputs.size(): (batch_size, text_len, 3)
        batch_size = inputs.size()[0]
        text_len = inputs.size()[1]
        
        outputs = []
        for i in range(text_len + 1):
            if i == 0:
                decoder_input = torch.tensor([self.start_symbol] * batch_size)
                hidden_state, cell = self._init_hidden_state(batch_size)
            else:
                decoder_input = inputs[:, i-1, :].unsqueeze(1)
            output, hidden_state, cell = self._forward_one_step(decoder_input, hidden_state, cell)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)  # outputs.size(): (batch_size, text_len + 1, 3, hidden_size * 2)

        return outputs


    def _forward_one_step(
            self,
            input,  # input.size(): (batch_size, 1, 3)
            hidden_state,  # hidden_state.size(): num_layers * 2, batch_size, hidden_size
            cell  # cell.size(): num_layers * 2, batch_size, hidden_size
        ):
        embedded_consonant = (self.dropout(self.embedding_consonant(input[:, :, 0].unsqueeze(-1)))).squeeze(2)  # embedded_consonant.size(): (batch_size, 1, embedding_size)
        embedded_vowel = (self.dropout(self.embedding_vowel(input[:, :, 1].unsqueeze(-1)))).squeeze(2) # embedded_vowel.size(): (batch_size, 1, embedding_size)
        embedded_tone = (self.dropout(self.embedding_tone(input[:, :, 2].unsqueeze(-1)))).squeeze(2)   # embedded_tone.size(): (batch_size, 1, embedding_size)
        embedding_output = torch.cat([embedded_consonant, embedded_vowel, embedded_tone], dim=1)  # embedding_outputs.size(): (batch_size, 3, embedding_size)

        output, (hidden_state, cell) = self.lstm(embedding_output, (hidden_state, cell))
        # output.size(): (batch_size, 3, 2 * hidden_size)
        # hidden_state.size(): (2 * num_layers, batch_size, hidden_size)
        # cell.size(): (2 * num_layers, batch_size, hidden_size)

        return output, hidden_state, cell

    def _init_hidden_state(self, batch_size):
        hidden_state, cell = (
            torch.zeros((self.num_layers * 2, batch_size, self.hidden_size)),
            torch.zeros((self.num_layers * 2, batch_size, self.hidden_size))
        )
        return hidden_state, cell


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

        self.linear_consonant = nn.Linear(
            in_features=hidden_size * 2,
            out_features=self.vocab.len()[0]
        )
        self.linear_vowel = nn.Linear(
            in_features=hidden_size * 2,
            out_features=self.vocab.len()[1]
        )
        self.linear_tone = nn.Linear(
            in_features=hidden_size * 2,
            out_features=self.vocab.len()[2]
        )

    def forward(
            self,
            outputs_encoder,  # outputs_encoder.size(): (batch_size, num_samples, hidden_size * 2)
            outputs_decoder,  # outputs_decoder.size(): (batch_size, text_len + 1, 3, hidden_size * 2)
        ):
        outputs_encoder = outputs_encoder.unsqueeze(2)  # outputs_encoder.size(): (batch_size, num_samples, 1, hidden_size * 2)

        consonants_decoder = outputs_decoder[:, :, 0, :]  # consonants_decoder.size(): (batch_size, text_len + 1, hidden_size * 2)
        vowels_decoder = outputs_decoder[:, :, 1, :]  # vowels_decoder.size(): (batch_size, text_len + 1, hidden_size * 2)
        tones_decoder = outputs_decoder[:, :, 2, :]  # tones_decoder.size(): (batch_size, text_len + 1, hidden_size * 2)

        consonants_decoder = consonants_decoder.unsqueeze(1)  # consonants_decoder.size(): (batch_size, 1, text_len + 1, hidden_size * 2)
        vowels_decoder = vowels_decoder.unsqueeze(1)  # vowels_decoder.size(): (batch_size, 1, text_len + 1, hidden_size * 2)
        tones_decoder = tones_decoder.unsqueeze(1)  # tones_decoder.size(): (batch_size, 1, text_len + 1, hidden_size * 2)

        outputs_consonant = self.join(outputs_encoder, consonants_decoder)  # outputs_consonant.size(): (batch_size, num_samples, text_len + 1, hidden_size * 2)
        outputs_vowel = self.join(outputs_encoder, vowels_decoder)  # outputs_vowel.size(): (batch_size,  num_samples, text_len + 1, hidden_size * 2)
        outputs_tone = self.join(outputs_encoder, tones_decoder)  # outputs_tone.size(): (batch_size, num_samples, text_len + 1, hidden_size * 2)

        outputs_consonant = self.linear_consonant(outputs_consonant)  # outputs_consonant.size(): (batch_size, num_samples, text_len + 1, num_consonants)
        outputs_vowel = self.linear_vowel(outputs_vowel)  # outputs_vowel.size(): (batch_size, num_samples, text_len + 1, num_vowels)
        outputs_tone = self.linear_tone(outputs_tone)  # outputs_tone.size(): (batch_size, num_samples, text_len + 1, num_tones)
        
        # outputs_consonant = F.softmax(outputs_consonant, dim=-1)  # outputs_consonant.size(): (batch_size, num_samples, text_len + 1, num_consonants)
        # outputs_vowel = F.softmax(outputs_vowel, dim=-1)  # outputs_vowel.size(): (batch_size, num_samples, text_len + 1, num_vowels)
        # outputs_tone = F.softmax(outputs_tone, dim=-1)  # outputs_tone.size(): (batch_size, num_samples, text_len + 1, num_tones)

        return outputs_consonant, outputs_vowel, outputs_tone


class RNNT(nn.Module):
    def __init__(self, input_size, hidden_size = 512, num_layers_encoder = 5, embedding_size = 1, num_layers_decoder = 2, dropout_prob = 0.2, mode = "additive"):
        super(RNNT, self).__init__()
        self.vocab = Vocabulary()
        self.encoder = Encoder(
            input_size,
            hidden_size = hidden_size,
            num_layers = num_layers_encoder,
            dropout_prob = dropout_prob
        )
        self.decoder = Decoder(
            vocab=self.vocab,
            embedding_size = embedding_size,
            hidden_size = hidden_size,
            num_layers = num_layers_decoder,
            dropout_prob = dropout_prob
        )
        self.joiner = Joiner(
            hidden_size=hidden_size,
            vocab=self.vocab,
            mode=mode
        )

    def compute_loss(
            self,
            inputs,
            signal_len,
            text_len,
            targets
        ):
        outputs_encoder = self.encoder(inputs)
        outputs_decoder = self.decoder(targets)
        outputs_joiner = self.joiner(outputs_encoder, outputs_decoder)

        consonant_joiner = outputs_joiner[0]  # consonant_joiner.size(): (batch_size, num_samples, text_len + 1, num_consonants)
        vowel_joiner = outputs_joiner[1]  # vowel_joiner.size(): (batch_size, num_samples, text_len + 1, num_vowels)
        tone_joiner = outputs_joiner[2]  # tone_joiner.size(): (batch_size, num_samples, text_len + 1, num_tones)

        consonant_target = targets[:, :, 0]
        vowel_target = targets[:, :, 1]
        tone_target = targets[:, :, 2]

        
        logit_lengths = torch.full((consonant_joiner.size(0),), consonant_joiner.size(1), dtype=torch.long)
        target_lengths = torch.full((consonant_joiner.size(0),), consonant_joiner.size(2), dtype=torch.long)
        
        print(consonant_target.size())
        print(signal_len)
        loss_consonant = rnnt_loss(
            logits=consonant_joiner,
            targets=consonant_target.int(),
            logit_lengths=logit_lengths.int(),
            target_lengths=target_lengths.int(),
            blank=self.vocab.get_index("<blank>")[0][0],
            reduction="mean"
        )
        loss_vowel = rnnt_loss(
            logits=vowel_joiner,
            targets=vowel_target.int(),
            logit_lengths=logit_lengths.int(),
            target_lengths=target_lengths.int(),
            blank=self.vocab.get_index("<blank>")[0][1],
            reduction="mean"
        )
        loss_tone = rnnt_loss(
            logits=tone_joiner,
            targets=tone_target.int(),
            logit_lengths=logit_lengths.int(),
            target_lengths=target_lengths.int(),
            blank=self.vocab.get_index("<blank>")[0][2],
            reduction="mean"
        )
        return loss_consonant, loss_vowel, loss_tone

    # def forward(self, inputs, max_len): => def decode?
    #     batch_size, num_samples, num_mels = inputs.size()

    #     outputs_encoder = self.encoder(inputs)  # outputs_encoder.size(): (batch_size, num_samples, hidden_size * 2)
    #     hidden_state, cell = self.decoder.init_hidden_state(batch_size)

    #     item = self._init_sos(batch_size)  # item.size(): (batch_size, 1, 3)
    #     counter = self._init_counter(batch_size)  # counter.size(): (batch_size)
        
    #     counter_ceil = num_samples - 1
    #     # term_state = torch.zeros(batch_size)
    #     t = 0

    #     while True:
    #         t += 1

    #         output_encoder = (outputs_encoder[range(batch_size), counter, :]).unsqueeze(1)  # signal.size(): (batch_size, 1, hidden_size * 2)
    #         prob_consonant, prob_vowel, prob_tone, hidden_state, cell = self._predict_next(item, output_encoder, hidden_state, cell)
    #         # prob_consonant.size(): (batch_size, num_consonants)
    #         # prob_vowel.size(): (batch_size, num_vowels)
    #         # prob_tone.size(): (batch_size, num_tones)
    #         # hidden_size.size(): (num_layers * 2, batch_size, hidden_size)
    #         # cell.size(): (num_layers * 2, batch_size, hidden_size)

    #         # prob_words = torch.cat((prob_consonant, prob_vowel, prob_tone), dim=1)
    #         # prob_words = prob_words.unsqueeze(1)
            
            
    #         predicted_word = torch.cat(
    #             (
    #                 torch.argmax(prob_consonant, dim=-1).unsqueeze(-1),  # .size(): (batch_size, 1)
    #                 torch.argmax(prob_vowel, dim=-1).unsqueeze(-1),  # .size(): (batch_size, 1)
    #                 torch.argmax(prob_tone, dim=-1).unsqueeze(-1)  # .size(): (batch_size, 1)
    #             ),
    #             dim=1
    #         )  # predicted_word.size(): (batch_size, 3)
    #         predicted_word = predicted_word.unsqueeze(1) # predicted_word.size(): (batch_size, 1, 3)

    #         if t == 1:
    #             # results = prob_words
    #             predictions = predicted_word  # predictions.size(): (batch_size, 1, 3)
    #         else:
    #             # results = torch.cat([results, prob_words], dim=1)
    #             predictions = torch.cat([predictions, predicted_word], dim=1)  # predictions.size(): (batch_size, t, 3)
                
    #         is_blank = self._is_blank(batch_size, predicted_word)  # is_blank.size(): (batch_size, 1)
    #         item = self._update_item(is_blank, item, predicted_word)  # item.size(): (batch_size, 1, 3)
            
    #         counter, update_mask = self._update(is_blank, counter, counter_ceil)

    #         if (update_mask.sum().item() == batch_size) or (t == max_len):  # counter tới max của tất cả trong batch hoặc đạt max len
    #             break
        
    #     return predictions  # predictions.size(): (batch_size, max_len, 3)

    # def _init_sos(self, batch_size):
    #     return torch.LongTensor(np.array([self.vocab.get_index("<sos>")] * batch_size))  # (batch_size, 1, 3)
    
    # def _init_counter(self, batch_size):
    #     return np.zeros(batch_size, dtype=int)
    
    # def _predict_next(
    #         self,
    #         item,  # item.size(): (batch_size, 1, 3)
    #         output_encoder,  # output_encoder.size(): (batch_size * num_samples, hidden_size * 2)
    #         hidden_state,
    #         cell
    #     ):
    #     output, hidden_state, cell = self.decoder(item, hidden_state, cell)
    #     prob_consonant, prob_vowel, prob_tone = self.joiner(output_encoder, output)
    #     return prob_consonant, prob_vowel, prob_tone, hidden_state, cell
    
    # def _is_blank(self, batch_size, predicted_word):
    #     is_blank = torch.zeros((batch_size, 1), dtype=bool)
    #     for i, predicted_word in enumerate(predicted_word):
    #         is_blank[i] = torch.equal(predicted_word, torch.LongTensor(self.vocab.get_index("<blank>")))
    #     return is_blank

    # def _update_item(self, is_blank, item, predicted_word):
    #     return ((is_blank * item.squeeze()) + (~is_blank * predicted_word.squeeze())).unsqueeze(1)
    
    # def _update(self, is_blank, counter, counter_ceil):
    #     counter = counter + is_blank.squeeze().numpy()
    #     counter, update_mask = self._clip_counter(counter, counter_ceil)
    #     # term_state = self._update_termination_state(term_state, update_mask, t)
    #     return counter, update_mask

    # def _clip_counter(self, counter, counter_ceil):
    #     update_mask = counter >= counter_ceil  # update_mask.shape: (batch_size,) (bool)
    #     upper_bounded = update_mask * counter_ceil  # upper_bounded.shape: (batch_size,)
    #     kept_counter = (counter < counter_ceil) * counter  # kept_counter.shape: (batch_size,)
    #     return upper_bounded + kept_counter, update_mask
    
    # # def _update_termination_state(self, term_state, update_mask, t):
    # #     is_unended = term_state == 0
    # #     to_update = is_unended & update_mask
    # #     return term_state + to_update * t
