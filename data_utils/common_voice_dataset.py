import torch
from torch.utils.data import Dataset
import torchaudio
import json
import os
from typing import Union

from data_utils.phoneme_vocab import PhonemeVocabv1, PhonemeVocabv2
from data_utils.phono_vocab import PhonoVocabv1, PhonoVocabv2
from data_utils.utils import normalize_script

def collate_fn(items: list[dict]) -> torch.Tensor:
    ids = [item["id"] for item in items]
    voices = [item["voice"] for item in items]
    scripts = [item["script"] for item in items]
    labels = [item["labels"] for item in items]

    # adding pad value for voice
    max_voice_len = max([voice.shape[-1] for voice in voices])
    for ith, voice in enumerate(voices):
        delta_len = max_voice_len - voice.shape[-1]
        pad_tensor = torch.zeros((1, delta_len)).float()
        voices[ith] = torch.cat([voice, pad_tensor], dim=-1)
    
    # adding pad value for encoded script
    max_label_len = max([label.shape[-1] for label in labels])
    for ith, label in enumerate(labels):
        delta_len = max_label_len - label.shape[-1]
        pad_tensor = torch.zeros((delta_len, )).float()
        labels[ith] = torch.cat([label, pad_tensor], dim=-1).unsqueeze(0)

    return {
        "id": ids,
        "voice": torch.tensor(voices),
        "script": scripts,
        "labels": torch.tensor(labels)
    }

class PhonemeCommonVoiceDataset(Dataset):
    def __init__(self, 
                 json_path: str, 
                 voice_path: str, 
                 sampling_rate: int,
                 vocab: Union[PhonemeVocabv1, PhonemeVocabv2, PhonoVocabv1, PhonoVocabv2]):
        super().__init__()

        self.voice_path = voice_path
        self.sampling_rate = sampling_rate
        self.vocab = vocab

        self._data: dict = json.load(open(json_path))
        self._keys = list(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        key = self._keys[idx]
        item = self._data[key]

        script = item["script"]
        script = script.lower()
        script = normalize_script(script)
        script_ids, word_indices = self.vocab.encode_script(script)

        audio_file = item["voice"]
        audio_file = audio_file.replace("mp3", "wav")
        voice, old_sampling_rate = torchaudio.load(os.path.join(self.voice_path, audio_file))
        voice = torchaudio.functional.resample(voice, orig_freq=old_sampling_rate, new_freq=self.sampling_rate)

        if not script == self.vocab.decode_script(script_ids, word_indices):
            print(script, "-", self.vocab.decode_script(script_ids, word_indices), "-", word_indices)

        return {
            "id": key,
            "voice": voice,
            "script": script,
            "labels": script_ids,
            "word_indices": word_indices
        }
