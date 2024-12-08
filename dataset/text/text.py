from torch.utils.data import Dataset
import json
from dataset.text.vocabulary import Vocabulary
import torch


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
        tokenized_script = self._tokenize(lowered_script)
        encoded_script = self._encode(tokenized_script)
        cut_script = self._cut_down_if_necessary(encoded_script)
        padded_script = self._pad_if_necessary(cut_script)
        return (torch.Tensor(padded_script)).long()
    
    def _get_script(self, index):
        return self.data[index][1]["script"]
    
    def _lower(self, script):
        return script.lower()

    def _tokenize(self, lowered_script):
        tokenized_script = []
        for word in lowered_script.split():
            if self.vocab._is_number(word):
                numbers = [number for number in word]
                for number in numbers:
                    tokenized_word = self.vocab._tokenize_word(number)
                    tokenized_script.append(tokenized_word)
            else:
                tokenized_word = self.vocab._tokenize_word(word)
                tokenized_script.append(tokenized_word)
        return tokenized_script
    
    def _encode(self, tokenized_script):
        encoded_script = []
        for tokenized_word in tokenized_script:
            encoded_word = self.vocab._encode_word(tokenized_word)
            encoded_script.append(encoded_word)
        return encoded_script
    
    def _cut_down_if_necessary(self, encoded_script):
        cut_script = encoded_script.copy()
        if len(encoded_script) > self.max_len:
            cut_script = encoded_script[:self.max_len]
        return cut_script
    
    def _pad_if_necessary(self, cut_script):
        padded_script = cut_script.copy()
        if len(cut_script) < self.max_len:
            pad_value = self.vocab._encode_word(self.vocab._tokenize_word("<pad>"))
            while len(padded_script) < self.max_len:
                padded_script.append(pad_value)
        return padded_script
