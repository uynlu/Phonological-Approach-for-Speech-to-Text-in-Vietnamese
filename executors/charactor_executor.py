import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler

from tqdm import tqdm
import os

from utils.instance import Instance, InstanceList
import evaluations


class CharactorExecutor:
    def __init__(
        self,
        model,
        device,
        vocab,
        learning_rate=3e-4,
        use_amp=False,
        weight_decay=0.00005
    ):
        self.model = model
        self.device = device
        self.vocab = vocab

        self.optim = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(.9, .98),
            eps=1e-05 if use_amp else 1e-09,
            weight_decay=weight_decay
        )

        self.grad_scaler = GradScaler(enabled=use_amp)
        self.loss = nn.CTCLoss(zero_infinity=True).to(device)
        self.epoch = 1
    
    def collate_fn(self, instances: list[Instance]) -> InstanceList:
        return InstanceList(instances, self.vocab.pad_idx)
    
    def create_dataloaders(self, train_dataset, dev_dataset, test_dataset, batch_size):
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn
        )
        self.dev_dataloader = DataLoader(
            dataset=dev_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn
        )
    
    def train(self):
        self.model.train()
        running_loss = .0
        with tqdm(desc='Epoch %d - Training' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for ith, items in enumerate(self.train_dataloader, start=1):
                items = items.to(self.device)
                inputs = items.voice.to(self.device)  # (batch_size, sequence_length, dim)
                input_lengths = torch.LongTensor(items.input_length).to(self.device)
                targets = items.labels.to(self.device)
                target_lengths = torch.LongTensor(items.target_length).to(self.device)

                outputs, output_lengths = self.model(inputs, input_lengths)
                loss = self.loss(outputs.transpose(0, 1), targets, output_lengths, target_lengths)

                self.optim.zero_grad()
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optim)
                self.grad_scaler.update()
                
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix({
                    "Loss": running_loss / ith
                })
                pbar.update()

        return running_loss / len(self.train_dataloader)

    def evaluate(self):
        self.model.eval()
        gen_scripts = []
        gt_scripts = []
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(self.dev_dataloader)) as pbar:
            for item in self.dev_dataloader:
                with torch.no_grad():
                    item = item.to(self.device)
                    inputs = item.voice.to(self.device)
                    input_length = torch.LongTensor(item.input_length).to(self.device)
                    target = item.script

                    outputs, _ = self.model(inputs, input_length)
                    predicted_ids = outputs.argmax(-1)

                gt_scripts.append(target[0])
                gen_scripts.append(self.vocab.decode_script(predicted_ids))
                
                pbar.update()
        
        scores = evaluations.compute_metrics(gt_scripts, gen_scripts)
        print("Evaluation scores on test: ", scores)
        

    def run(self, convergence_threshold=0.0001, loss_threshold=0.01):
        prev_loss = float('inf')
        count = 0
        
        while True:
            current_loss = self.train()
            self.evaluate()

            if abs(prev_loss - current_loss) < convergence_threshold:
                count += 1
            else:
                count = 0

            if count >= 5 and current_loss < loss_threshold:
                break

            prev_loss = current_loss

            # self.save_checkpoint()

            self.epoch += 1

    def get_prediction(self):
        self.model.eval()
        gen_scripts = []
        gt_scripts = []
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(self.test_dataloader)) as pbar:
            for item in self.test_dataloader:
                with torch.no_grad():
                    item = item.to(self.device)
                    inputs = item.voice.to(self.device)
                    input_length = torch.LongTensor(item.input_length).to(self.device)
                    target = item.script

                    outputs, _ = self.model(inputs, input_length)
                    predicted_ids = outputs.argmax(-1)

                gt_scripts.append(target[0])
                gen_scripts.append(self.vocab.decode_script(predicted_ids))
                
                pbar.update()
        print(f"predicted_scripts:{gt_scripts}")
        print(f"predicted_scripts:{gen_scripts}")

        scores = evaluations.compute_metrics(gt_scripts, gen_scripts)
        print("Evaluation scores on test: ", scores)

    def save_checkpoint(self, dict_for_updating):
        pass

    def load_checkpoint(self, fname):
        pass