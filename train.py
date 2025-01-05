from torch.optim import Adam
from tqdm import tqdm
import torch


class Trainer:
    def __init__(self, model, lr, vocab, device):
        self.model = model
        self.lr = lr
        self.optimizer = Adam(model.parameters(), lr=self.lr)
        self.vocab = vocab
        self.device = device
  
    def train(self, train_loader, max_len, print_interval = 20):
        train_loss = 0
        num_samples = 0
        self.model.train()
        for index, batch in enumerate(tqdm(train_loader)):
            (signals, signal_lens), (scripts, text_lens) = batch
            signals = signals.to(self.device)
            signal_lens = signal_lens.to(self.device)
            scripts = scripts.to(self.device)
            text_lens = text_lens.to(self.device)
            batch_size = signals.size()[0]
            num_samples += batch_size
            loss = self.model.compute_loss(signals, signal_lens, text_lens, scripts)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * batch_size

            if index % print_interval == 0:
                self.model.eval()
                guesses = self.model.greedy_search(signals, signal_lens, max_len)
                print("\n")
                for b in range(2):
                    print("guess:", self.vocab.decode_script(guesses[b]))
                    print("truth:", self.vocab.decode_script(scripts[b, :text_lens[b]]))

        train_loss /= num_samples
        
        return train_loss

    def test(self, dev_loader, max_len, print_interval=10):
        test_loss = 0
        num_samples = 0
        self.model.eval()
        with torch.no_grad():
            for idx, batch in tqdm(dev_loader):
                (signals, signal_lens), (scripts, text_lens) = batch
                signals = signals.to(self.device)
                signal_lens = signal_lens.to(self.device)
                scripts = scripts.to(self.device)
                text_lens = text_lens.to(self.device)
                batch_size = signals.size()[0]
                num_samples += batch_size
                loss = self.model.compute_loss(signals, signal_lens, text_lens, scripts)
                test_loss += loss.item() * batch_size
                if idx % print_interval == 0:
                    print("\n")
                    print("guess:", self.vocab.decode_script(self.model.greedy_search(signals, signal_lens, max_len)[0]))
                    print("truth:", self.vocab.decode_script(scripts[0, :text_lens[0]]))
                    print("")
        test_loss /= num_samples
        return test_loss
