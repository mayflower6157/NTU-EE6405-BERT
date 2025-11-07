import pickle
from loguru import logger
import random
import typing
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast
from datasets import Dataset as HFDataset
from utils.device_utils import get_device


class CacheLoadError(RuntimeError):
    """Raised when an on-disk cache cannot be deserialized."""


class IMDBBertDataset(Dataset):
    MASK_PERCENTAGE = 0.15
    OPTIMAL_LENGTH_PERCENTILE = 70
    MAX_LENGTH = 512  # BERT limit

    def __init__(
        self,
        path,
        ds_from=None,
        ds_to=None,
        should_include_text=False,
        device=None,
        cache_dir=None,
        overwrite_cache=False,
    ):
        self.device = device or get_device()
        self.should_include_text = should_include_text
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        # --- cache setup ---
        cache_dir = Path(cache_dir or Path(path).parent / "cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"imdb_cached_{ds_from or 0}_{ds_to or 'end'}.pt"
        self.cache_path = cache_dir / cache_name

        # --- try to load cache ---
        cache_loaded = False
        if self.cache_path.exists() and not overwrite_cache:
            logger.info(f"🔁 Loading cached dataset from {self.cache_path}")
            try:
                self.df, self.columns = self._safe_load_cache(self.cache_path)
                cache_loaded = True
            except CacheLoadError as err:
                logger.warning(
                    "Cache at %s is unreadable (%s). Rebuilding...",
                    self.cache_path,
                    err,
                )

        if not cache_loaded:
            logger.info(f"⚙️ Processing dataset from {path}")
            self.ds: pd.Series = pd.read_csv(path)["review"]
            if ds_from is not None or ds_to is not None:
                self.ds = self.ds[ds_from:ds_to]

            self.columns = (
                [
                    "masked_ids",
                    "masked_sentence",
                    "raw_text",
                    "token_mask",
                    "target_indices",
                    "attention_mask",
                    "is_next",
                ]
                if should_include_text
                else [
                    "masked_ids",
                    "token_mask",
                    "target_indices",
                    "attention_mask",
                    "is_next",
                ]
            )

            self.df = self.prepare_dataset()
            self._save_cache()

    def _save_cache(self):
        payload = {
            "version": 2,
            "columns": self.columns,
            "data": {col: self.df[col].tolist() for col in self.df.columns},
        }
        torch.save(
            payload,
            self.cache_path,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
            _use_new_zipfile_serialization=False,
        )
        logger.success(f"✅ Cached dataset at {self.cache_path}")

    # ---------------------------------------------------------------------- #
    # safe cache load compatible with PyTorch 2.6+
    def _safe_load_cache(self, path: Path):
        import pandas as pd

        try:
            block_manager = pd.core.internals.managers.BlockManager
        except AttributeError:
            block_manager = None

        safe_globals = [pd.DataFrame, pd.Series, pd.Index]
        if block_manager is not None:
            safe_globals.append(block_manager)
        torch.serialization.add_safe_globals(safe_globals)

        try:
            try:
                cached = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                cached = torch.load(path, map_location="cpu")
        except (EOFError, RuntimeError, pickle.UnpicklingError) as err:
            raise CacheLoadError("torch.load failed") from err

        columns = cached.get("columns")

        if "data" in cached:
            df = pd.DataFrame(cached["data"], columns=columns)
            return df, columns

        if "df" in cached:  # legacy cache that stored raw DataFrame
            df = cached["df"]
            columns = columns or list(df.columns)
            logger.info("♻️ Rewriting legacy cache with safe serialization format...")
            self._rewrite_legacy_cache(path, df, columns)
            return df, columns

        raise CacheLoadError(f"Unsupported cache format in {path}")

    def _rewrite_legacy_cache(self, path: Path, df: pd.DataFrame, columns):
        safe_payload = {
            "version": 2,
            "columns": columns,
            "data": {col: df[col].tolist() for col in columns},
        }
        torch.save(
            safe_payload,
            path,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
            _use_new_zipfile_serialization=False,
        )
        logger.success(f"Legacy cache upgraded at {path}")

    # ---------------------------------------------------------------------- #
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        input_ids = torch.tensor(item["masked_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.bool)
        mlm_mask = torch.tensor(item["token_mask"], dtype=torch.bool)
        labels = torch.tensor(item["target_indices"], dtype=torch.long)
        labels = labels.clone()
        labels[~mlm_mask] = 0  # ignore unmasked

        nsp_target = torch.tensor([1, 0] if item["is_next"] == 0 else [0, 1], dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_mask": mlm_mask,
            "labels": labels,
            "next_sentence_label": nsp_target,
        }

    # ---------------------------------------------------------------------- #
    def prepare_dataset(self) -> pd.DataFrame:
        sentences, nsp, sentence_lens = [], [], []

        for review in self.ds:
            parts = review.split(". ")
            sentences += parts
            self._update_length(parts, sentence_lens)
        self.optimal_sentence_length = min(
            self.MAX_LENGTH, self._find_optimal_sentence_length(sentence_lens)
        )

        logger.info("🧩 Preprocessing dataset...")
        for review in tqdm(self.ds, total=len(self.ds)):
            parts = review.split(". ")
            if len(parts) > 1:
                for i in range(len(parts) - 1):
                    nsp.append(self._create_item(parts[i], parts[i + 1], 1))
                    f, s = self._select_false_nsp_sentences(sentences)
                    nsp.append(self._create_item(f, s, 0))

        return pd.DataFrame(nsp, columns=self.columns)

    # ---------------------------------------------------------------------- #
    def _update_length(self, sentences: typing.List[str], lengths: typing.List[int]):
        for v in sentences:
            lengths.append(len(v.split()))
        return lengths

    def _find_optimal_sentence_length(self, lengths: typing.List[int]):
        return int(np.percentile(np.array(lengths), self.OPTIMAL_LENGTH_PERCENTILE))

    def _select_false_nsp_sentences(self, sentences: typing.List[str]):
        s_len = len(sentences)
        s1 = random.randint(0, s_len - 1)
        s2 = random.randint(0, s_len - 1)
        while s2 == s1 + 1:
            s2 = random.randint(0, s_len - 1)
        return sentences[s1], sentences[s2]

    # ---------------------------------------------------------------------- #
    def _create_item(self, first: str, second: str, is_next: int):
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
        special_mask = encoding["special_tokens_mask"]

        masked_ids, token_mask, targets = self._mask_tokens(input_ids, special_mask, attention_mask)

        if self.should_include_text:
            return (
                masked_ids,
                [t for t, m in zip(self.tokenizer.convert_ids_to_tokens(masked_ids), attention_mask) if m],
                [t for t, m in zip(self.tokenizer.convert_ids_to_tokens(input_ids), attention_mask) if m],
                token_mask,
                targets,
                attention_mask,
                is_next,
            )
        return masked_ids, token_mask, targets, attention_mask, is_next

    # ---------------------------------------------------------------------- #
    def _mask_tokens(self, input_ids, special_mask, attention_mask):
        token_mask = [True] * len(input_ids)
        target_indices = input_ids.copy()
        specials = set(self.tokenizer.all_special_ids)
        candidates = [
            i
            for i, (tid, sm) in enumerate(zip(input_ids, special_mask))
            if sm == 0 and tid not in specials
        ]
        n_mask = round(len(candidates) * self.MASK_PERCENTAGE)
        mask_idx = random.sample(candidates, min(n_mask, len(candidates)))
        masked_ids = input_ids.copy()

        for i in mask_idx:
            r = random.random()
            if r < 0.8:
                masked_ids[i] = self.tokenizer.mask_token_id
            elif r < 0.9:
                while True:
                    rid = random.randint(0, self.tokenizer.vocab_size - 1)
                    if rid not in specials:
                        masked_ids[i] = rid
                        break
            token_mask[i] = False

        return masked_ids, token_mask, target_indices

    # ---------------------------------------------------------------------- #
    def to_hf_dataset(self):
        logger.info("📦 Converting to Hugging Face Dataset...")
        recs = {
            "input_ids": self.df["masked_ids"].tolist(),
            "attention_mask": self.df["attention_mask"].tolist(),
            "token_mask": self.df["token_mask"].tolist(),
            "labels": self.df["target_indices"].tolist(),
            "next_sentence_label": self.df["is_next"].tolist(),
        }
        ds = HFDataset.from_dict(recs)
        ds.set_format(type="torch", columns=list(recs.keys()))
        logger.success(f"✅ Hugging Face dataset ready with {len(ds)} samples.")
        return ds


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    ds = IMDBBertDataset(
        BASE / "data/imdb.csv",
        ds_from=0,
        ds_to=5000,
        should_include_text=True,
        cache_dir=BASE / "data/cache",
    )
    hf_ds = ds.to_hf_dataset()
    logger.info("Sample 0:")
    for k, v in hf_ds[0].items():
        val = v.tolist() if torch.is_tensor(v) else v
        print(f"{k}: {val[:20] if hasattr(val, '__getitem__') else val}")
    print("\nReadable text:")
    print(ds.tokenizer.decode(hf_ds[0]['input_ids'], skip_special_tokens=True))
