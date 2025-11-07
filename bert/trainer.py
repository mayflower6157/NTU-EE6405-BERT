import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.device_utils import get_device


class BertTrainer:
    """Trainer for BERT-like models with MLM+NSP, warmup, cosine scheduling,
    gradient accumulation, clipping, and TensorBoard logging."""

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
        scheduler_type="linear",  # or "cosine"
        grad_accum_steps=1,
        max_grad_norm=1.0,
        device=None,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        self.grad_accum_steps = grad_accum_steps
        self.scheduler_type = scheduler_type

        # TensorBoard setup
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.loss_history = defaultdict(list)
        self.epoch_history = defaultdict(list)

        # Optimization setup
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.nsp_criterion = nn.BCEWithLogitsLoss()

        # Learning rate scheduling
        steps_per_epoch = math.ceil(len(self.loader) / grad_accum_steps)
        self.total_steps = steps_per_epoch * epochs
        self.warmup_steps = max(1, int(self.total_steps * warmup_ratio))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        # Mixed precision setup (for A5000 efficiency)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ---------------------------------------------------------
    # Learning rate schedule
    # ---------------------------------------------------------
    def _lr_lambda(self, step: int) -> float:
        """Linear or cosine LR decay with warmup."""
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))

        progress = (step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        if self.scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)

    def _compute_losses(self, outputs, batch) -> Dict[str, torch.Tensor]:
        token_logits, nsp_logits = outputs
        token_logits = token_logits.transpose(1, 2)
        mlm_loss = self.mlm_criterion(token_logits, batch["labels"].to(self.device))
        nsp_loss = self.nsp_criterion(
            nsp_logits, batch["next_sentence_label"].float().to(self.device)
        )
        return {"mlm": mlm_loss, "nsp": nsp_loss, "total": mlm_loss + nsp_loss}

    def _record_step_losses(self, global_step: int, losses: Dict[str, torch.Tensor]):
        for name, loss in losses.items():
            value = loss.item()
            self.loss_history[name].append(value)
            self.writer.add_scalar(f"Step/{name}", value, global_step)

    def _record_epoch_losses(self, epoch: int, losses: Dict[str, float]):
        for name, value in losses.items():
            self.epoch_history[name].append(value)
            self.writer.add_scalar(f"Epoch/{name}", value, epoch + 1)
        self.writer.flush()

    def _one_epoch(self, epoch: int, global_step_start: int) -> Tuple[int, Dict[str, float]]:
        self.model.train()
        loop = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
        global_step = global_step_start
        epoch_totals = defaultdict(float)
        batch_count = 0

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
            batch_count += 1
            self._record_step_losses(global_step, losses)
            for name, loss in losses.items():
                epoch_totals[name] += loss.item()

            loop.set_postfix(
                {
                    "Loss": f"{losses['total'].item():.4f}",
                    "MLM": f"{losses['mlm'].item():.4f}",
                    "NSP": f"{losses['nsp'].item():.4f}",
                    "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )
        epoch_means = {
            name: total / max(1, batch_count) for name, total in epoch_totals.items()
        }
        self._record_epoch_losses(epoch, epoch_means)
        return global_step, epoch_means

    def train(self):
        print(f"Training on {self.device} for {self.epochs} epochs...")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        global_step = 0
        for epoch in range(self.epochs):
            start = time.time()
            global_step, epoch_losses = self._one_epoch(epoch, global_step)
            print(
                f"Epoch {epoch+1}/{self.epochs} completed in {time.time()-start:.2f}s"
            )
            if epoch_losses:
                print(
                    " | ".join(
                        [
                            f"{name.upper()} avg: {value:.4f}"
                            for name, value in epoch_losses.items()
                        ]
                    )
                )
            self._save_checkpoint(epoch)
        self._save_loss_plot()
        self.writer.close()

    def _save_checkpoint(self, epoch: int):
        path = self.checkpoint_dir / f"bert_epoch_{epoch+1}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved at {path}")

    def _save_loss_plot(self):
        if not self.loss_history["total"]:
            return
        steps = range(1, len(self.loss_history["total"]) + 1)
        plt.figure(figsize=(10, 5))
        for name in ["total", "mlm", "nsp"]:
            if self.loss_history[name]:
                plt.plot(steps, self.loss_history[name], label=name.upper())
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("BERT Training Losses")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plot_path = self.log_dir / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved loss curve to {plot_path}")
