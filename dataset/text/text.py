from torch.utils.data import Dataset
import json
from dataset.text.vocabulary import PhonemeVocabv2
# import re
# import string
# import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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
        # script = self.lower(script)
        # script = self.remove_special_characters(script)
        # script = self.tokenize(script)
        script = self.encode(script)
        script = self.cut_down_if_necessary(script)
        script = self.add_token(script)
        len_script = script.shape[0]

        script = self.pad_if_necessary(script)
        
        return script, len_script
    
    def get_script(self, index):
        return self.data[index][1]["script"]
    
    # def lower(self, script):
    #     lowered_script = script.lower()
    #     return lowered_script
    
    # def remove_special_characters(self, script):
    #     pattern = f"[{re.escape(string.punctuation)}]"
    #     removed_script = re.sub(pattern, "", script) 
    #     return removed_script

    # def tokenize(self, script):
    #     tokenized_script = self.vocab.tokenize(script.split()[0])
    #     for word in script.split()[1:]:
    #         tokenized_word = self.vocab.tokenize(word)
    #         tokenized_script = np.vstack((tokenized_script, tokenized_word))
    #     return tokenized_script
    
    def encode(self, script):
        encoded_script = self.vocab.encode_script(script)
        return encoded_script
    
    def cut_down_if_necessary(self, script):
        cut_script = script.copy()
        if len(script) >= self.max_len - 2:
            cut_script = script[:self.max_len - 2]
        return cut_script
    
    def add_token(self, script):
        added_script = np.vstack((self.vocab.bos_idx, script, self.vocab.eos_idx))
        return added_script
    
    def pad_if_necessary(self, script):
        padded_script = script.copy()
        if len(script) < self.max_len:
            pad_value = self.vocab.pad_idx
            while len(padded_script) < self.max_len:
                padded_script = np.vstack((padded_script, pad_value))
        return padded_script
    