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
        script = self._lower(script)
        script = self._tokenize(script)
        script = self._encode(script)
        script = self._cut_down_if_necessary(script)
        script = self._pad_if_necessary(script)
        return torch.Tensor(script)
    
    def _get_script(self, index):
        return self.data[index][1]["script"]
    
    def _lower(self, script):
        return script.lower()

    def _tokenize(self, script):
        tokenized_script = []
        for word in script.split():
            tokenized_word = self.vocab._tokenize_word(word)
            tokenized_script.append(tokenized_word)
        return tokenized_script
    
    def _encode(self, script):
        encoded_script = []
        for tokenized_word in script:
            encoded_word = self.vocab._encode_word(tokenized_word)
            encoded_script.append(encoded_word)
        return encoded_script
    
    def _cut_down_if_necessary(self, script):
        if len(script) > self.max_len:
            script = script[:self.max_len]
        return script
    
    def _pad_if_necessary(self, script):
        while len(script) < self.max_len:
            script.append(self.vocab._encode_word(self.vocab._tokenize_word("<pad>")))
        return script
    