from torch.utils.data import Dataset
import json
from dataset.text.vocabulary import Vocabulary
import re
import string
import torch
import numpy as np


class CustomText(Dataset):
    def __init__(self, data_directory, max_len):
        super(CustomText, self).__init__()

        with open(data_directory, mode="r", encoding="utf-8") as file:
            data = json.load(file)
        self.data = list(data.items())

        self.vocab = Vocabulary()
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        script = self._get_script(index)
        lowered_script = self._lower(script)
        removed_script = self._remove_special_characters(lowered_script)
        tokenized_script = self._tokenize(removed_script)
        encoded_script = self._encode(tokenized_script)
        cut_script = self._cut_down_if_necessary(encoded_script)
        padded_script = self._pad_if_necessary(cut_script)
        final_script = self._add_token(padded_script)
        return torch.tensor(final_script, dtype=torch.long)
    
    def _get_script(self, index):
        return self.data[index][1]["script"]
    
    def _lower(self, script):
        lowered_script = script.lower()
        return lowered_script
    
    def _remove_special_characters(self, lowered_script):
        pattern = f"[{re.escape(string.punctuation)}]"
        removed_script = re.sub(pattern, "", lowered_script) 
        return removed_script

    def _tokenize(self, removed_script):
        tokenized_script = self.vocab.tokenize(removed_script.split()[0])
        for word in removed_script.split()[1:]:
            tokenized_word = self.vocab.tokenize(word)
            tokenized_script = np.vstack((tokenized_script, tokenized_word))
        return tokenized_script
    
    def _encode(self, tokenized_script):
        encoded_script = self.vocab.encode_word(tokenized_script[0])
        for tokenized_word in tokenized_script[1:]:
            encoded_word = self.vocab.encode_word(tokenized_word)
            encoded_script = np.vstack((encoded_script, encoded_word))
        return encoded_script
    
    def _cut_down_if_necessary(self, encoded_script):
        cut_script = encoded_script.copy()
        if len(encoded_script) > self.max_len:
            cut_script = encoded_script[:self.max_len]
        return cut_script
    
    def _pad_if_necessary(self, cut_script):
        padded_script = cut_script.copy()
        if len(cut_script) < self.max_len:
            pad_value = self.vocab.get_index("<pad>")
            while len(padded_script) < self.max_len:
                padded_script = np.vstack((padded_script, pad_value))
        return padded_script
    
    def _add_token(self, padded_script):
        final_script = np.array(padded_script)
        final_script = np.vstack((self.vocab.get_index("<sos>"), padded_script, self.vocab.get_index("<eos>")))
        return final_script
    