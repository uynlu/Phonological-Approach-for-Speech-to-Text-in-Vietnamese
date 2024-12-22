
from torch.utils.data import Dataset
import json
from dataset.text.vocabulary import PhonemeVocabv2
# import re
# import string
import torch
# import numpy as np

class CustomText(Dataset):
    def __init__(self, data_directory, max_len):
        super(CustomText, self).__init__()

        with open(data_directory, mode="r", encoding="utf-8") as file:
            data = json.load(file)
        self.data = list(data.items())

        self.vocab = PhonemeVocabv2()
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        script = self.get_script(index)
        script = self.encode(script)
        script = self.cut_down_if_necessary(script)
        script = self.add_token(script)
        script = self.pad_if_necessary(script)
        return script
    
    def get_script(self, index):
        return self.data[index][1]["script"]
    
    # def _lower(self, script):
    #     lowered_script = script.lower()
    #     return lowered_script
    
    # def _remove_special_characters(self, lowered_script):
    #     pattern = f"[{re.escape(string.punctuation)}]"
    #     removed_script = re.sub(pattern, "", lowered_script) 
    #     return removed_script

    # def _tokenize(self, removed_script):
    #     tokenized_script = self.vocab.tokenize(removed_script.split()[0])
    #     for word in removed_script.split()[1:]:
    #         tokenized_word = self.vocab.tokenize(word)
    #         tokenized_script = np.vstack((tokenized_script, tokenized_word))
    #     return tokenized_script
    
    def encode(self, script):
        return self.vocab.encode_script(script)
    
    def cut_down_if_necessary(self, script):
        if len(script) > self.max_len - 2:
            script = script[:self.max_len]
        return script
    
    def add_token(self, script):
        return torch.cat((torch.tensor([self.vocab.bos_idx]), script, torch.tensor([self.vocab.eos_idx])))
    
    def pad_if_necessary(self, script):
        pad_value = torch.tensor([self.vocab.pad_idx])
        while len(script) < self.max_len:
            script = torch.cat((script, pad_value))
        return script
    