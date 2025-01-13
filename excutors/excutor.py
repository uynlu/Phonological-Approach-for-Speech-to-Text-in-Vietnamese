import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
import os
import numpy as np

from utils.instance import Instance, InstanceList
import evaluations


class Excutor:
    def __init__(
        self,
        model,
        device,
        vocab,
        learning_rate=0.1,
        use_amp=False,
        weight_decay=0.00005,
        checkpoint_path="checkpoints",
        warmup=4000,
        d_model=512
    ):
        self.model = model
        self.device = device
        self.vocab = vocab
        self.warmup = warmup
        self.d_model = d_model

        self.optim = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(.9, .98),
            eps=1e-05 if use_amp else 1e-09,
            weight_decay=weight_decay
        )

        self.scheduler = LambdaLR(self.optim, self.lambda_lr)
        self.grad_scaler = GradScaler(enabled=use_amp)
        self.loss = nn.CTCLoss(zero_infinity=True).to(device)
        self.epoch = 1
        self.checkpoint_path = checkpoint_path
        
        os.makedirs(self.checkpoint_path, exist_ok=True)
    
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
        
            print(gt_scripts)
            print(gen_scripts)
        
        scores= evaluations.compute_metrics(gt_scripts, gen_scripts)
        print("Evaluation scores on test: %s", scores)


    def run(self, convergence_threshold=0.001, loss_threshold=0.1):
        # checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
        # if checkpoint:
        #     self.epoch = checkpoint["epoch"] + 1  

        prev_loss = float('inf')  
        count = 0  

        while True:
            current_loss = self.train()
            self.evaluate()
            
            print('current loss:', current_loss)
            if current_loss < loss_threshold:
                break

            if abs(prev_loss - current_loss) < convergence_threshold:
                count += 1
            else:
                count = 0  

            if count >= 5:  
                break

            prev_loss = current_loss

            # self.save_checkpoint()

            self.epoch += 1


    def get_predictions(self):
        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
        results = {}
        for item in self.test_dataloader:
            with torch.no_grad():
                item = item.to(self.device)
                inputs = item.voice.to(self.device)
                input_length = torch.LongTensor(item.input_length).to(self.device)
                target = item.script

                outputs, _ = self.model(inputs, input_length)
                predicted_ids = outputs.argmax(-1)

            gen_script = self.vocab.decode_script(predicted_ids)
            gt_script = item.script
            id = item.id[0]
            results[id] = {
                "prediction": gen_script[0],
                "reference": gt_script[0]
            }
            
        predictions = [results[id]["prediction"] for id in results]
        references = [results[id]["reference"] for id in results]

        scores = evaluations.compute_metrics(references, predictions)
        
    # def lambda_lr(self, step):
    #     warm_up = self.warmup
    #     step += 1
    #     return (self.d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)

    # def load_checkpoint(self, fname) -> dict:
    #     if not os.path.exists(fname):
    #         return None

    #     checkpoint = torch.load(fname, map_location=self.device)

    #     self.model.load_state_dict(checkpoint['state_dict'], strict=False)

    #     if 'optimizer' in checkpoint:
    #         self.optim.load_state_dict(checkpoint['optimizer'])
    #     if 'scheduler' in checkpoint and hasattr(self, 'scheduler'):
    #         self.scheduler.load_state_dict(checkpoint['scheduler'])

    #     return checkpoint

    # def save_checkpoint(self, dict_for_updating: dict = None) -> None:
    #     dict_for_saving = {
    #         'epoch': self.epoch,
    #         'state_dict': self.model.state_dict(),
    #         'optimizer': self.optim.state_dict(),
    #         'scheduler': self.scheduler.state_dict()
    #     }

    #     if dict_for_updating:
    #         dict_for_saving.update(dict_for_updating)

    #     save_path = os.path.join(self.checkpoint_path, "last_model.pth")
    #     torch.save(dict_for_saving, save_path)
