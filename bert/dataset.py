from loguru import logger
import random
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast
from datasets import Dataset as HFDataset
from utils.device_utils import get_device


class IMDBBertDataset(Dataset):
    MASK_PERCENTAGE = 0.15  # How much of the words to mask
    OPTIMAL_LENGTH_PERCENTILE = 70
    MAX_LENGTH = 512  # BERT's maximum sequence length

    def __init__(
        self, path, ds_from=None, ds_to=None, should_include_text=False, device=None
    ):
        self.ds: pd.Series = pd.read_csv(path)["review"]
        if ds_from is not None or ds_to is not None:
            self.ds = self.ds[ds_from:ds_to]

        # Use pretrained BERT tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        self.optimal_sentence_length = None
        self.device = device or get_device()
        self.should_include_text = should_include_text

        if should_include_text:
            self.columns = [
                "masked_ids",
                "masked_sentence",
                "raw_text",
                "token_mask",
                "target_indices",
                "attention_mask",
                "is_next",
            ]
        else:
            self.columns = [
                "masked_ids",
                "token_mask",
                "target_indices",
                "attention_mask",
                "is_next",
            ]

        self.df = self.prepare_dataset()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # Convert to tensors
        input_ids = torch.tensor(item["masked_ids"], dtype=torch.long)
        attention_mask = torch.tensor(
            item["attention_mask"], dtype=torch.bool
        )  # <- Bool
        mlm_mask = torch.tensor(item["token_mask"], dtype=torch.bool)  # <- Bool
        labels = torch.tensor(item["target_indices"], dtype=torch.long)

        # very important: ignore unmasked positions for MLM loss
        labels = labels.clone()
        labels[~mlm_mask] = 0  # 0 must match ignore_index in CrossEntropyLoss

        nsp_target = torch.tensor(
            [1, 0] if item["is_next"] == 0 else [0, 1], dtype=torch.float
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_mask": mlm_mask,
            "labels": labels,
            "next_sentence_label": nsp_target,
        }

    def prepare_dataset(self) -> pd.DataFrame:
        sentences, nsp = [], []
        sentence_lens = []

        # Split dataset into sentences
        for review in self.ds:
            review_sentences = review.split(". ")
            sentences += review_sentences
            self._update_length(review_sentences, sentence_lens)
        self.optimal_sentence_length = min(
            self.MAX_LENGTH, self._find_optimal_sentence_length(sentence_lens)
        )

        logger.info("Preprocessing dataset...")
        for review in tqdm(self.ds):
            review_sentences = review.split(". ")
            if len(review_sentences) > 1:
                for i in range(len(review_sentences) - 1):
                    # True NSP
                    first, second = review_sentences[i], review_sentences[i + 1]
                    nsp.append(self._create_item(first, second, 1))

                    # False NSP
                    first, second = self._select_false_nsp_sentences(sentences)
                    nsp.append(self._create_item(first, second, 0))

        df = pd.DataFrame(nsp, columns=self.columns)
        return df

    def _update_length(self, sentences: typing.List[str], lengths: typing.List[int]):
        for v in sentences:
            l = len(v.split())
            lengths.append(l)
        return lengths

    def _find_optimal_sentence_length(self, lengths: typing.List[int]):
        arr = np.array(lengths)
        return int(np.percentile(arr, self.OPTIMAL_LENGTH_PERCENTILE))

    def _select_false_nsp_sentences(self, sentences: typing.List[str]):
        sentences_len = len(sentences)
        s1 = random.randint(0, sentences_len - 1)
        s2 = random.randint(0, sentences_len - 1)
        while s2 == s1 + 1:
            s2 = random.randint(0, sentences_len - 1)
        return sentences[s1], sentences[s2]

    def _create_item(self, first: str, second: str, is_next: int):
        # Use tokenizer to handle truncation and padding to the model limit
        encoding = self.tokenizer(
            first,
            second,
            max_length=self.MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        special_tokens_mask = encoding["special_tokens_mask"]

        # Convert IDs back to tokens for optional inspection/debugging (strip padding)
        tokens = [
            token
            for token, mask in zip(
                self.tokenizer.convert_ids_to_tokens(input_ids), attention_mask
            )
            if mask == 1
        ]

        # Mask some tokens for MLM
        masked_token_ids, token_mask, target_indices = self._mask_tokens(
            input_ids, special_tokens_mask, attention_mask
        )

        masked_sentence_text = [
            tok
            for tok, m in zip(
                self.tokenizer.convert_ids_to_tokens(masked_token_ids), attention_mask
            )
            if m == 1
        ]

        if self.should_include_text:
            return (
                masked_token_ids,
                masked_sentence_text,
                tokens,
                token_mask,
                target_indices,
                attention_mask,
                is_next,
            )
        else:
            return masked_token_ids, token_mask, target_indices, attention_mask, is_next

    def _mask_tokens(
        self,
        input_ids: typing.List[int],
        special_tokens_mask: typing.List[int],
        attention_mask: typing.List[int],
    ):
        len_s = len(input_ids)
        token_mask = [True] * len_s
        target_indices = input_ids.copy()

        special_token_ids = set(self.tokenizer.all_special_ids)
        # Eligible positions exclude special tokens and padding
        candidate_indices = [
            idx
            for idx, (token_id, special_flag) in enumerate(
                zip(input_ids, special_tokens_mask)
            )
            if special_flag == 0 and token_id not in special_token_ids
        ]

        mask_amount = round(len(candidate_indices) * self.MASK_PERCENTAGE)

        mask_indices = random.sample(
            candidate_indices, min(mask_amount, len(candidate_indices))
        )

        masked_token_ids = input_ids.copy()

        for i in mask_indices:
            prob = random.random()
            if prob < 0.8:
                # 80% Replace with [MASK]
                masked_token_ids[i] = self.tokenizer.mask_token_id
            elif prob < 0.9:
                # 10% Replace with random token (excluding special tokens)
                while True:
                    rand_token_id = random.randint(0, self.tokenizer.vocab_size - 1)
                    if rand_token_id not in special_token_ids:
                        break
                masked_token_ids[i] = rand_token_id
            else:
                # 10% Keep original token (no change)
                pass
            token_mask[i] = False

        return masked_token_ids, token_mask, target_indices

    def to_hf_dataset(self):
        """Convert pandas DataFrame to Hugging Face Dataset efficiently."""
        logger.info("Converting to Hugging Face Dataset...")

        # Use stored attention_mask directly
        masked_ids = self.df["masked_ids"].tolist()
        attention_masks = self.df["attention_mask"].tolist()

        # Create records dictionary with attention masks included
        records = {
            "input_ids": masked_ids,
            "attention_mask": attention_masks,
            "token_mask": self.df["token_mask"].tolist(),
            "labels": self.df["target_indices"].tolist(),
            "next_sentence_label": self.df["is_next"].tolist(),
        }

        # Create HuggingFace dataset from prepared records
        hf_dataset = HFDataset.from_dict(records)

        # Set tensor format
        hf_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "token_mask",
                "labels",
                "next_sentence_label",
            ],
        )

        logger.success(f"Dataset ready with {len(hf_dataset)} samples.")
        return hf_dataset


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    ds = IMDBBertDataset(
        BASE_DIR.joinpath("data/imdb.csv"),
        ds_from=0,
        ds_to=5000,
        should_include_text=True,
    )

    # Convert to Hugging Face Dataset
    hf_ds = ds.to_hf_dataset()

    # Show structure
    logger.info("Hugging Face dataset preview:")
    logger.info(hf_ds)

    # Display a few sample records (pretty printed)
    logger.info("Sample record 0:")
    for k, v in hf_ds[0].items():
        val = v.tolist() if torch.is_tensor(v) else v
        print(f"{k}: {val[:20] if hasattr(val, '__getitem__') else val}")

    print("\nReadable text:")
    print(ds.tokenizer.decode(hf_ds[0]["input_ids"], skip_special_tokens=True))
