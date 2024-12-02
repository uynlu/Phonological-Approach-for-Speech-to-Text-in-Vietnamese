from torch.utils.data import Dataset
import json
import re
from dataset.text.vocabulary import Vocabulary


class CustomText(Dataset):
    def __init__(self, data_directory):
        super(CustomText, self).__init__()

        with open(data_directory, mode="r", encoding="utf-8") as file:
            data = json.load(file)
        self.data = list(data.items())

        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        script = self._get_script(index)
        lowered_script = self._lower(script)
        tokenized_script = self._tokenize(lowered_script)
        encoded_script = self._encode(tokenized_script)
        return encoded_script
    
    def _get_script(self, index):
        return self.data[index][1]["script"]
    
    def _lower(self, script):
        return script.lower()

    def _tokenize(self, lowered_script):
        tokenized_script = []
        for word in lowered_script.split():
            tokenized_word = self.vocab._tokenize_word(word)
            tokenized_script.append(tokenized_word)
        return tokenized_script
    
    def _encode(self, tokenized_script):
        encoded_script = []
        for tokenized_word in tokenized_script:
            encoded_word = self.vocab._encode_word(tokenized_word)
            encoded_script.append(encoded_word)
        return encoded_script
 