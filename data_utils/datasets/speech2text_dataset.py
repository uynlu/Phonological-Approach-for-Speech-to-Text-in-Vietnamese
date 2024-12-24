from torch.utils.data import Dataset
import torchaudio
import json
import os

from builders.dataset_builder import META_DATASET
from data_utils.vocabs.character_vocab import CharacterVocab
from data_utils.utils import normalize_script
from utils.instance import Instance
from data_utils.utils import MelSpectrogram

@META_DATASET.register()
class CharacterDataset(Dataset):
    def __init__(self, config, vocab: CharacterVocab):
        super().__init__()

        self.voice_path = config.voice_path
        self.vocab = vocab
        self.transformer = MelSpectrogram(config.transformer)

        self._data: dict = json.load(open(config.json_path))
        self._keys = list(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        key = self._keys[idx]
        item = self._data[key]

        script = item["script"]
        script = script.lower()
        script = normalize_script(script)
        script_ids = self.vocab.encode_script(script)

        audio_file = item["voice"]
        audio_file = audio_file.replace("mp3", "wav")
        voice, _ = torchaudio.load(os.path.join(self.voice_path, audio_file))
        voice = self.transformer.transform(voice)
        voice = voice.squeeze(0).transpose(-1, -2)

        return Instance(
            id = key,
            voice = voice,
            script = script,
            labels = script_ids
        )

@META_DATASET.register()
class PhonemeDataset(CharacterDataset):
    def __getitem__(self, idx: int) -> dict:
        key = self._keys[idx]
        item = self._data[key]

        script = item["script"]
        script = script.lower()
        script = normalize_script(script)
        script_ids, word_indices = self.vocab.encode_script(script)

        audio_file = item["voice"]
        audio_file = audio_file.replace("mp3", "wav")
        voice, _ = torchaudio.load(os.path.join(self.voice_path, audio_file))
        voice = self.transformer.transform(voice)
        voice = voice.squeeze(0).transpose(-1, -2)

        return Instance(
            id = key,
            voice = voice,
            script = script,
            labels = script_ids,
            word_indices = word_indices
        )
