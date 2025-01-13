import torch

from data_utils.vocabs.character_vocab import CharacterVocab
from data_utils.datasets.speech2text_dataset import CharacterDataset
from models.conformer.model import Conformer
from excutors import Excutor
from torch.utils.data import ConcatDataset

if __name__ == "__main__":
    vocab = CharacterVocab()
    train_commonvoice = CharacterDataset(
        "/kaggle/working/common_voice/wav-voices",
        16000,
        80,
        400,
        512,
        1024,
        "/kaggle/working/common_voice/train.json",
        vocab
    )
    dev_commonvoice = CharacterDataset(
        "/kaggle/working/common_voice/wav-voices",
        16000,
        80,
        400,
        512,
        1024,
        "/kaggle/working/common_voice/dev.json",
        vocab
    )
    test_commonvoice = CharacterDataset(
        "/kaggle/working/common_voice/wav-voices",
        16000,
        80,
        400,
        512,
        1024,
        "/kaggle/working/common_voice/test.json",
        vocab
    )
    train_vivos = CharacterDataset(
        "/kaggle/working/vivos/voices",
        16000,
        80,
        400,
        512,
        1024,
        "/kaggle/working/vivos/train.json",
        vocab
    )
    test_vivos = CharacterDataset(
        "/kaggle/working/vivos/voices",
        16000,
        80,
        400,
        512,
        1024,
        "/kaggle/working/vivos/test.json",
        vocab
    )
    dataset = ConcatDataset([train_commonvoice, dev_commonvoice, test_commonvoice, train_vivos, test_vivos])
    train, test = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train, dev = torch.utils.data.random_split(train, [0.8, 0.2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conformer(
        num_classes=vocab.size, 
        input_dim=80, 
        encoder_dim=32,
        num_encoder_layers=3
    ).to(device)
    excutor = Excutor(
        model=model,
        device=device,
        vocab=vocab,
    )
    excutor.create_dataloaders(train, dev, test, 64)
    excutor.run()
    