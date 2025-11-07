import math
import time
from pathlib import Path
from typing import Dict
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.device_utils import get_device


class BertTrainer:
    """Trainer for BERT-like models with MLM+NSP, warmup, clipping, and tqdm."""

    def __init__(
        self,
        model,
        dataset,
        log_dir,
        checkpoint_dir,
        batch_size=16,
        learning_rate=1e-5,
        epochs=5,
        warmup_ratio=0.06,
        max_grad_norm=1.0,
        device=None,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.nsp_criterion = nn.BCEWithLogitsLoss()

        steps_per_epoch = math.ceil(len(self.loader))
        self.total_steps = steps_per_epoch * epochs
        self.warmup_steps = max(1, int(self.total_steps * warmup_ratio))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.total_steps - step)
            / float(max(1, self.total_steps - self.warmup_steps)),
        )

    def _compute_losses(self, outputs, batch) -> Dict[str, torch.Tensor]:
        token_logits, nsp_logits = outputs
        token_logits = token_logits.transpose(1, 2)
        mlm_loss = self.mlm_criterion(token_logits, batch["labels"].to(self.device))
        nsp_loss = self.nsp_criterion(
            nsp_logits, batch["next_sentence_label"].float().to(self.device)
        )
        return {"mlm": mlm_loss, "nsp": nsp_loss, "total": mlm_loss + nsp_loss}

    def _one_epoch(self, epoch: int, global_step_start: int) -> int:
        self.model.train()
        loop = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
        global_step = global_step_start

        for batch in loop:
            global_step += 1
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )
            losses = self._compute_losses(outputs, batch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            loop.set_postfix(
                {
                    "Loss": f"{losses['total'].item():.4f}",
                    "MLM": f"{losses['mlm'].item():.4f}",
                    "NSP": f"{losses['nsp'].item():.4f}",
                    "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )
        return global_step

    def train(self):
        print(f"Training on {self.device} for {self.epochs} epochs...")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        global_step = 0
        for epoch in range(self.epochs):
            start = time.time()
            global_step = self._one_epoch(epoch, global_step)
            print(
                f"Epoch {epoch+1}/{self.epochs} completed in {time.time()-start:.2f}s"
            )
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch: int):
        path = self.checkpoint_dir / f"bert_epoch_{epoch+1}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved at {path}")
