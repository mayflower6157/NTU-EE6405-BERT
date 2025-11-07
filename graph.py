import torch

from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bert.dataset import IMDBBertDataset
from bert.model import BERT
from utils.device_utils import get_device

BASE_DIR = Path(__file__).resolve().parent

EMB_SIZE = 64
HIDDEN_SIZE = 36
BATCH_SIZE = 12
NUM_HEADS = 4
LOG_DIR = BASE_DIR / "data" / "logs" / "graph"

device = get_device()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


def main():
    logger.info("Preparing tiny dataset for graph export")
    dataset = IMDBBertDataset(
        BASE_DIR / "data" / "imdb.csv", ds_from=0, ds_to=5, device=device
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    vocab_size = dataset.tokenizer.vocab_size
    model = BERT(vocab_size, EMB_SIZE, HIDDEN_SIZE, NUM_HEADS, device=device)
    writer = SummaryWriter(str(LOG_DIR))

    batch = next(iter(loader))
    inputs = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    writer.add_graph(model, input_to_model=[inputs, attention_mask])
    logger.success("Saved model graph to {}", LOG_DIR)


if __name__ == "__main__":
    main()
