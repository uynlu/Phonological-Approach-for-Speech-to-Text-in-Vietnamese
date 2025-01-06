import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tqdm import tqdm

from builders.executor_builder import META_EXECUTOR
from executors.base_executor import BaseExecutor
from utils.logging_utils import setup_logger
import evaluations

logger = setup_logger()

@META_EXECUTOR.register()
class CharacterBasedExecutor(BaseExecutor):
    def __init__(self, config):
        super().__init__(config)

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=(.9, .98),
            eps=1e-05 if config.training.use_amp else 1e-09,
            weight_decay=config.training.weight_decay
            )
        self.grad_scaler = torch.amp.GradScaler(enabled=config.training.use_amp)

    def configuring_hyperparameters(self, config):
        self.epoch = 1
        self.warmup = config.training.warmup
        self.score = config.training.score
        self.learning_rate = config.training.learning_rate
        self.patience = config.training.patience

    def evaluate_metrics(self, dataloader: DataLoader):
        self.model.eval()
        gen_scripts = []
        gt_scripts = []
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for item in dataloader:
                with torch.no_grad():
                    item = item.to(self.device)
                    predicted_ids = self.model.generate(item)

                gt_scripts.append(item.script[0])
                gen_scripts.append(self.vocab.decode_script(predicted_ids)[0])
                
                print(gt_scripts)
                print(gen_scripts)
                pbar.update()

        scores= evaluations.compute_metrics(gt_scripts, gen_scripts)

        return scores

    def train(self):
        self.model.train()
        running_loss = .0
        with tqdm(desc='Epoch %d - Training' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for ith, items in enumerate(self.train_dataloader, start=1):
                self.scheduler.step()
                # forward pass
                items = items.to(self.device)
                _, loss = self.model(items)

                # backward pass
                self.optim.zero_grad()
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optim)
                self.grad_scaler.update()

                # update the training status
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix({
                    "Loss": running_loss / ith
                })
                pbar.update()
