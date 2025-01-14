import torch
from torch.utils.data import ConcatDataset

from data_utils.vocabs.phoneme_vocab import PhonemeVocabv2
from data_utils.datasets.speech2text_dataset import PhonemeDataset
from models.conformer.model import Conformer
from executors.phoneme_executor import PhonemeExecutor


if __name__ == "__main__":
    vocab = PhonemeVocabv2()
    train_commonvoice = PhonemeDataset(
        "/kaggle/working/common_voice/wav-voices",
        16000,
        80,
        400,
        512,
        1024,
        "/kaggle/working/common_voice/train.json",
        vocab
    )
    dev_commonvoice = PhonemeDataset(
        "/kaggle/working/common_voice/wav-voices",
        16000,
        80,
        400,
        512,
        1024,
        "/kaggle/working/common_voice/dev.json",
        vocab
    )
    test_commonvoice = PhonemeDataset(
        "/kaggle/working/common_voice/wav-voices",
        16000,
        80,
        400,
        512,
        1024,
        "/kaggle/working/common_voice/test.json",
        vocab
    )
    train_vivos = PhonemeDataset(
        "/kaggle/working/vivos/voices",
        16000,
        80,
        400,
        512,
        1024,
        "/kaggle/working/vivos/train.json",
        vocab
    )
    test_vivos = PhonemeDataset(
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
    excutor = PhonemeExecutor(
        model=model,
        device=device,
        vocab=vocab,
    )
    excutor.create_dataloaders(train, dev, test, 64)
    excutor.run()
    excutor.get_prediction()
    