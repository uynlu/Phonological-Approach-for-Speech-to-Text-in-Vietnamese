import torch
import torch.nn as nn
from speechbrain.nnet.loss.transducer_loss import TransducerLoss
from model.rnnt.encoder import Encoder
from model.rnnt.decoder import Decoder
from model.rnnt.joiner import Joiner


class RNNT(nn.Module):
    def __init__(self, vocab, input_size, device, hidden_size = 512, num_layers_encoder = 5, num_layers_decoder = 2, dropout_prob = 0.2, mode = "additive"):
        super(RNNT, self).__init__()
        self.vocab = vocab
        self.device = device

        self.encoder = Encoder(
            input_size,
            hidden_size = hidden_size,
            num_layers = num_layers_encoder,
            dropout_prob = dropout_prob
        ).to(self.device)
        self.decoder = Decoder(
            vocab=self.vocab,
            hidden_size = hidden_size,
            num_layers = num_layers_decoder,
            dropout_prob = dropout_prob
        ).to(self.device)
        self.joiner = Joiner(
            hidden_size=hidden_size,
            vocab=self.vocab,
            mode=mode
        ).to(self.device)

    def compute_loss(
            self,
            inputs,
            signal_len,
            text_len,
            targets
        ):
        outputs_encoder = self.encoder(inputs)
        outputs_decoder = self.decoder.forward(targets)
        outputs_joiner = self.joiner(outputs_encoder, outputs_decoder)

        transducer_loss = TransducerLoss(self.vocab.blank_idx)
        loss = transducer_loss(outputs_joiner, targets, signal_len, text_len)
        return loss

    def greedy_decode(self, signals, signal_lens, max_len):
        outputs_batch = []
        batch_size = signals.size()[0]
        outputs_encoder = self.encoder(signals)
        for b in range(batch_size):
            t = 0
            u = 0
            outputs = [self.decoder.start_symbol]
            hidden_state = self.decoder.initial_state.unsqueeze(0)
            while t < signal_lens[b] and u < max_len:
                decoder_input = torch.tensor([outputs[-1]], device = self.device)
                output, hidden_state = self.decoder.forward_one_step(decoder_input, hidden_state)
                feature_t = outputs_encoder[b, t]
                output_joiner = self.joiner.forward(feature_t, output)
                argmax = output_joiner.max(-1)[1].item()
                if argmax == self.vocab.blank_idx:
                    t += 1
                else: # argmax == a label
                    u += 1
                    outputs.append(argmax)
            outputs_batch.append(outputs[1:-1])
        return outputs_batch

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
