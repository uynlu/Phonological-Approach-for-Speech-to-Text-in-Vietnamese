from torch.utils.data import Dataset
from dataset.speech.speech import CustomSpeech
from dataset.text.text import CustomText


class CustomDataset(Dataset):
    def __init__(self, audio_directory, data_directory, target_sample_rate, num_samples, transformation=False):
        super(CustomDataset, self).__init__()

        self.speech = CustomSpeech(audio_directory, data_directory, target_sample_rate, num_samples, transformation=False)
        self.script = CustomText(data_directory) 
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.speech[index], self.script[index]
