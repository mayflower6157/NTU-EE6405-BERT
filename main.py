import datetime
from pathlib import Path
from loguru import logger

from bert.dataset import IMDBBertDataset
from bert.model import BERT
from bert.trainer import BertTrainer
from utils.device_utils import get_device

# ==========================================================
# ⚙️ Global configuration
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "imdb.csv"
CHECKPOINT_DIR = BASE_DIR / "data" / "bert_checkpoints"

# Model hyperparameters
EMB_SIZE = 128          # Larger embedding dimension for full dataset
HIDDEN_SIZE = 64        # Deeper hidden projection layer
NUM_HEADS = 8           # 8 attention heads = more expressive self-attention
BATCH_SIZE = 32         # Fits cleanly on RTX A5000 (24 GB VRAM)
ACCUM_STEPS = 4         # Gradient accumulation → effective batch = 128
EPOCHS = 1
LEARNING_RATE = 5e-5    # Base LR for cosine schedule
WARMUP_RATIO = 0.1      # Warmup = 10% of total steps

# Logging directory
timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
LOG_DIR = BASE_DIR / "data" / "logs" / f"bert_experiment_{timestamp}"

# ==========================================================
# 🚀 Main training function
# ==========================================================
def main():
    device = get_device()
    logger.success(f"Using device: {device}")

    # ------------------------------------------------------
    # 🧩 Dataset preparation
    # ------------------------------------------------------
    logger.info(f"Preparing full IMDB dataset from {DATA_PATH}")
    dataset = IMDBBertDataset(
        DATA_PATH,
        ds_from=0,
        ds_to=None,     # ✅ Use full dataset
        device=device,
    )
    logger.info(f"Loaded IMDB dataset with {len(dataset)} samples")

    # ------------------------------------------------------
    # 🧠 Model initialization
    # ------------------------------------------------------
    vocab_size = dataset.tokenizer.vocab_size
    model = BERT(
        vocab_size=vocab_size,
        dim_inp=EMB_SIZE,
        dim_out=HIDDEN_SIZE,
        attention_heads=NUM_HEADS,
        device=device,
    )
    logger.info(f"Model initialized with vocab size {vocab_size}")

    # ------------------------------------------------------
    # 🧰 Trainer setup (includes cosine LR + accumulation)
    # ------------------------------------------------------
    trainer = BertTrainer(
        model=model,
        dataset=dataset,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        grad_accum_steps=ACCUM_STEPS,
        warmup_ratio=WARMUP_RATIO,
        scheduler_type="cosine",  # "linear" or "cosine"
    )

    # ------------------------------------------------------
    # 🏋️ Start training
    # ------------------------------------------------------
    logger.info(
        f"Starting full training on IMDB (batch={BATCH_SIZE}, accum={ACCUM_STEPS}, heads={NUM_HEADS})"
    )
    trainer.train()

    logger.success(f"Training complete! TensorBoard logs at: {LOG_DIR}")


if __name__ == "__main__":
    main()