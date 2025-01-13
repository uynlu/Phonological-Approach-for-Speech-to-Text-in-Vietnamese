from torch.utils.data import Dataset
import torchaudio
import json
import os

from data_utils.vocabs.character_vocab import CharacterVocab
from data_utils.utils import normalize_script
from utils.instance import Instance
from data_utils.utils import MelSpectrogram


class CharacterDataset(Dataset):
    def __init__(
            self,
            voice_path,
            sampling_rate,
            n_mels,
            win_length,
            hop_length,
            n_ffts,
            json_path,
            vocab: CharacterVocab
        ):
        super().__init__()

        self.voice_path = voice_path
        self.vocab = vocab
        self.transformer = MelSpectrogram(sampling_rate, n_mels, win_length, hop_length, n_ffts)

        self._data: dict = json.load(open(json_path, encoding="utf8"))
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
        # shifted_right_script_ids = script_ids[1:]
        # script_ids = script_ids[:-1]

        audio_file = item["voice"]
        audio_file = audio_file.replace("mp3", "wav")
        voice, _ = torchaudio.load(os.path.join(self.voice_path, audio_file))
        voice = self.transformer.transform(voice)
        voice = voice.squeeze(0).transpose(0, 1)

        input_length = voice.shape[0]
        target_length = script_ids.shape[0]

        return Instance(
            id = key,
            voice = voice,
            input_length=input_length,
            target_length=target_length,
            script = script,
            labels = script_ids
        )


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

        input_length = voice.shape[0]
        target_length = script_ids.shape[0]

        return Instance(
            id = key,
            voice = voice,
            input_length=input_length,
            target_length=target_length,
            script = script,
            labels = script_ids,
            word_indices = word_indices
        )
