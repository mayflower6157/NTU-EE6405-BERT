import datetime
from pathlib import Path

from loguru import logger

from bert.dataset import IMDBBertDataset
from bert.model import BERT
from bert.trainer import BertTrainer
from utils.device_utils import get_device

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "imdb.csv"
CHECKPOINT_DIR = BASE_DIR / "data" / "bert_checkpoints"

EMB_SIZE = 64
HIDDEN_SIZE = 36
NUM_HEADS = 4
BATCH_SIZE = 12
EPOCHS = 15
LEARNING_RATE = 7e-5

timestamp = datetime.datetime.utcnow().timestamp()
LOG_DIR = BASE_DIR / "data" / "logs" / f"bert_experiment_{timestamp}"


def main():
    device = get_device()
    logger.info("Preparing dataset from {}", DATA_PATH)
    dataset = IMDBBertDataset(
        DATA_PATH,
        ds_from=0,
        ds_to=1000,
        device=device,
    )

    vocab_size = dataset.tokenizer.vocab_size
    logger.info("Initializing model with vocab size {}", vocab_size)
    model = BERT(vocab_size, EMB_SIZE, HIDDEN_SIZE, NUM_HEADS, device=device)

    trainer = BertTrainer(
        model=model,
        dataset=dataset,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
