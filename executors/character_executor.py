import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
from shutil import copyfile
import json
from tqdm import tqdm

from builders.executor_builder import META_EXECUTOR
from executors.base_executor import BaseExecutor
from utils.logging_utils import setup_logger
import evaluations

logger = setup_logger()

@META_EXECUTOR.register()
class CharacterBasedExecutor(BaseExecutor):
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
                with torch.no_grad:
                    item = item.to(self.device)
                    predicted_ids = self.model.generate(item.voice)

                gt_scripts.append(item.script)
                gen_scripts.append(self.vocab.decode_script(predicted_ids))
                
                pbar.update()

        scores= evaluations.compute_metrics(gt_scripts, gen_scripts)

        return scores

    def train(self):
        self.model.train()
        running_loss = .0
        with tqdm(desc='Epoch %d - Training' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for ith, items in enumerate(self.train_dataloader, start=1):
                # forward pass
                items = items.to(self.device)
                _, loss = self.model(items.voice, items.labels)

                # backward pass
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # update the training status
                this_loss = loss.item()
                running_loss += this_loss

                self.scheduler.step()

                pbar.set_postfix({
                    "Loss": running_loss / ith
                })
                pbar.update()
    
    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            best_val_score = checkpoint["best_val_score"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"] + 1
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            best_val_score = .0
            patience = 0

        while True:
            self.train()

            # val scores
            scores = self.evaluate_metrics(self.dev_dataloader)
            logger.info("Validation scores %s", scores)
            val_score = scores[self.score]

            # Prepare for next epoch
            best = False
            if val_score > best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            exit_train = False

            if patience == self.patience:
                logger.info('patience reached.')
                exit_train = True

            self.save_checkpoint({
                'best_val_score': best_val_score,
                'patience': patience
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path, "last_model.pth"), 
                        os.path.join(self.checkpoint_path, "best_model.pth"))

            if exit_train:
                break

            self.epoch += 1

    def get_predictions(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            self.logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
        results = {}
        self.logger.info(f"Epoch {self.epoch+1} - Evaluating")
        for item in self.test_dataloader:
            with torch.no_grad():
                item = item.to(self.device)
                predicted_ids = self.model.generate(item.voice)

            gen_script = self.vocab.decode_script(predicted_ids)
            gt_script = item.script
            id = item.id[0]
            results[id] = {
                "prediction": gen_script,
                "reference": gt_script
            }

        predictions = [results[id]["prediction"] for id in results]
        references = [results[id]["reference"] for id in results]
        scores, _ = evaluations.compute_metrics(references, predictions)
        self.logger.info("Evaluation scores on test: %s", scores)

        json.dump({
            "results": results,
            **scores,
        }, open(os.path.join(self.checkpoint_path, "test_results.json"), "w+"), ensure_ascii=False)

