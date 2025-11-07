import math
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class BertTrainer:
    """High-performance BERT trainer with AMP, compile, warmup, accumulation,
    TensorBoard logging, auto-resume, and checkpoint cleanup."""

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
        scheduler_type="cosine",  # "linear" or "cosine"
        grad_accum_steps=1,
        max_grad_norm=1.0,
        device=None,
        num_workers=8,
        resume=True,
        keep_last_n_checkpoints=3,
    ):
        # Set precision BEFORE any CUDA operations to avoid warnings
        if torch.cuda.is_available():
            # New API for controlling TF32 precision (PyTorch 2.9+)
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            torch.backends.cudnn.benchmark = True
        
        self.device = device
        self.model = model.to(self.device)

        try:
            self.model = torch.compile(self.model)
        except Exception as e:
            print(f"[⚠️] torch.compile not supported or failed: {e}")

        # Optimized DataLoader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
        )

        self.epochs = epochs
        self.resume = resume
        self.keep_last_n = keep_last_n_checkpoints

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging setup
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.loss_history = defaultdict(list)
        self.epoch_history = defaultdict(list)

        # Optimization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.nsp_criterion = nn.BCEWithLogitsLoss()

        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.scheduler_type = scheduler_type

        steps_per_epoch = math.ceil(len(self.loader) / grad_accum_steps)
        self.total_steps = steps_per_epoch * epochs
        self.warmup_steps = max(1, int(self.total_steps * warmup_ratio))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        # AMP Scaler - use the NEW API
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        # Auto-resume setup
        self.start_epoch = 1
        if self.resume:
            self._maybe_resume_checkpoint()

    # ---------------------------------------------------------
    # LR Scheduler (Linear or Cosine)
    # ---------------------------------------------------------
    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = (step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        if self.scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)

    # ---------------------------------------------------------
    # Resume from checkpoint if available
    # ---------------------------------------------------------
    def _maybe_resume_checkpoint(self):
        ckpts = sorted(self.checkpoint_dir.glob("bert_epoch_*.pt"))
        if not ckpts:
            return
        latest_ckpt = ckpts[-1]
        print(f"🔁 Resuming from checkpoint: {latest_ckpt.name}")
        checkpoint = torch.load(latest_ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        print(f"✅ Successfully restored model from epoch {checkpoint['epoch']}")

    # ---------------------------------------------------------
    # Clean old checkpoints to save space
    # ---------------------------------------------------------
    def _cleanup_old_checkpoints(self):
        ckpts = sorted(self.checkpoint_dir.glob("bert_epoch_*.pt"))
        if len(ckpts) > self.keep_last_n:
            old_ckpts = ckpts[:-self.keep_last_n]
            for ckpt in old_ckpts:
                try:
                    ckpt.unlink()
                    print(f"🧹 Deleted old checkpoint: {ckpt.name}")
                except Exception as e:
                    print(f"[⚠️] Could not delete {ckpt.name}: {e}")

    # ---------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------
    def train(self):
        global_step = 0
        self.model.train()

        tqdm.write(f"\n🚀 Training on {self.device} for {self.epochs} epochs...\n")

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_loss, mlm_loss_total, nsp_loss_total = 0.0, 0.0, 0.0
            pbar = tqdm(
                self.loader,
                desc=f"Epoch {epoch}/{self.epochs}"
                f" | loss=0.0000 | MLM=0.0000 | NSP=0.0000",
                leave=False,
            )

            for step, batch in enumerate(pbar):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    token_pred, cls_pred = self.model(
                        batch["input_ids"], batch["attention_mask"]
                    )
                    loss_mlm = self.mlm_criterion(
                        token_pred.view(-1, token_pred.size(-1)),
                        batch["labels"].view(-1),
                    )
                    loss_nsp = self.nsp_criterion(
                        cls_pred.squeeze(-1), batch["next_sentence_label"].float()
                    )
                    loss = (loss_mlm + loss_nsp) / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    global_step += 1

                epoch_loss += loss.item() * self.grad_accum_steps
                mlm_loss_total += loss_mlm.item()
                nsp_loss_total += loss_nsp.item()

                if (step + 1) % 10 == 0 or (step + 1) == len(self.loader):
                    pbar.set_description(
                        f"Epoch {epoch}/{self.epochs}"
                        f" | Loss={epoch_loss / (step + 1):.4f}"
                        f" | MLM={mlm_loss_total / (step + 1):.4f}"
                        f" | NSP={nsp_loss_total / (step + 1):.4f}"
                        f" | LR={self.optimizer.param_groups[0]['lr']:.2e}"
                    )

            avg_loss = epoch_loss / len(self.loader)
            avg_mlm = mlm_loss_total / len(self.loader)
            avg_nsp = nsp_loss_total / len(self.loader)
            self.epoch_history["total"].append(avg_loss)
            self.epoch_history["mlm"].append(avg_mlm)
            self.epoch_history["nsp"].append(avg_nsp)
            self.writer.add_scalars("Epoch", {"total": avg_loss, "mlm": avg_mlm, "nsp": avg_nsp}, epoch)

            ckpt_path = self.checkpoint_dir / f"bert_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                },
                ckpt_path,
            )

            tqdm.write(
                f"✅ Epoch {epoch}/{self.epochs} completed | "
                f"Loss={avg_loss:.4f}, MLM={avg_mlm:.4f}, NSP={avg_nsp:.4f}"
            )

            # Clean up older checkpoints
            self._cleanup_old_checkpoints()

        self.writer.close()
        tqdm.write(f"🏁 Training complete! Checkpoints → {self.checkpoint_dir}\n")