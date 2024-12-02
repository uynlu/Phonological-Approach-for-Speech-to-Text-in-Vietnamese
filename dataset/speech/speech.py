import torch.nn as nn
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
import json
import os
import torch
import torch.nn.functional as F


class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=64, win_length=400, hop_length=512, n_ffts=1024):
        # sample_rate: tần suất lấy mẫu (16000 điểm/s)
        # n_mels: chiều cao của Mel Spectrogram?
        # win_length: (nếu chiều dài của data là 800 => 2 window)
        # hop_length: bước nhảy (stride)
        # n_ffts: chiều dài mỗi Time-Section

        super(MelSpectrogram, self).__init__()

        self.transform = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels, win_length=win_length,
            hop_length=hop_length,
            n_fft=n_ffts
        )

    def forward(self, input):
        return self.transform(input)


class MFCC(nn.Module):
    def __init__(self, sample_rate=16000, n_mfcc=50, n_mels=64, win_length=400, hop_length=512, n_ffts=1024):
        super(MFCC, self).__init__()
        
        self.transform = transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_mels": n_mels,
                "n_fft": n_ffts,
                "hop_length": hop_length
            }
        )

    def forward(self, input):
        return self.transform(input)


class CustomSpeech(Dataset):
    def __init__(self, audio_directory, data_directory, target_sample_rate, num_samples, transformation):
        super(CustomSpeech, self).__init__()

        self.audio_directory = audio_directory

        with open(data_directory, mode="r", encoding="utf-8") as file:
            data = json.load(file)
        self.data = list(data.items())

        if transformation == "Mel Spectrogram":
            self.transformation = MelSpectrogram()
        elif transformation == "MFCC":
            self.transformation = MFCC()
        
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        audio_wave, sample_rate = torchaudio.load(audio_path)  # torch.Tensor(num_channel, origin_num_sample), int
        signal = self._resample_if_necessary(audio_wave, sample_rate)  # torch.Tensor(resampled_num_sample, N)
        signal = self._cut_down_if_necessary(signal)
        signal = self._pad_if_necessary(signal)  # torch.Tensor(num_channel, num_sample)
        mel = self.transformation(signal)  # torch.Tensor(num_channel, num_mels, transformed_num_sample)
        mel = mel.squeeze()
        mel = mel.permute(1, 0)  # torch.Tensor(transformed_num_sample, num_mels)
        return mel

    def _get_audio_path(self, index):
        file = (self.data[index][1]['voice']).replace(".mp3", ".wav")
        return os.path.join(self.audio_directory, file)
    
    # Không phải tất cả mẫu đều có sample_rate như nhau => resample
    def _resample_if_necessary(self, audio_wave, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = transforms.Resample(sample_rate, self.target_sample_rate)
            audio_wave = resampler(audio_wave)
        return audio_wave
    
    def _cut_down_if_necessary(self, signal):
        if signal.size()[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _pad_if_necessary(self, signal):
        length_signal = signal.size()[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        return signal
    