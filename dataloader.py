import logging
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Tuple, Optional
from transformers import PreTrainedTokenizerFast
from model import Llama4TextConfig
from trainer import TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class TinyStoriesDataset(IterableDataset, Dataset):
    """Dataset for TinyStories, supporting streaming and non-streaming modes."""
    
    def __init__(
        self,
        seq_len: int = 128,
        is_streaming: bool = True,
        device: str = 'cuda',
        prefetch_size: int = 4096,
        token_file_path: Optional[str] = None,
        token_ids: Optional[torch.Tensor] = None,
        vocab_size: int = 10000
    ):
        """Initialize the TinyStories dataset.
        
        Args:
            seq_len (int): Sequence length for input and target. Defaults to 128.
            is_streaming (bool): Whether to use streaming mode. Defaults to True.
            device (str): Device to move data to ('cuda' or 'cpu'). Defaults to 'cuda'.
            prefetch_size (int): Number of samples to prefetch in streaming mode. Defaults to 4096.
            token_file_path (Optional[str]): Path to tokenized data file. Defaults to None.
            token_ids (Optional[torch.Tensor]): Pre-loaded token IDs. Defaults to None.
            vocab_size (int): Vocabulary size for token validation. Defaults to 10000.
        
        Raises:
            ValueError: If neither token_file_path nor token_ids is provided, token length is too short,
                        or invalid token IDs are found.
        """
        super().__init__()
        self.seq_len = seq_len
        self.is_streaming = is_streaming
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.prefetch_size = prefetch_size
        self.vocab_size = vocab_size

        if token_ids is not None:
            self.token_ids = token_ids
        elif token_file_path is not None:
            try:
                self.token_ids = torch.load(token_file_path, map_location='cpu')
            except Exception as e:
                raise ValueError(f"Failed to load token file {token_file_path}: {e}")
        else:
            raise ValueError("Either 'token_file_path' or 'token_ids' must be provided.")

        # Validate and fix token IDs
        if isinstance(self.token_ids, torch.Tensor):
            invalid_tokens = (self.token_ids < 0) | (self.token_ids >= self.vocab_size)
            if invalid_tokens.any():
                invalid_indices = torch.where(invalid_tokens)[0]
                invalid_values = self.token_ids[invalid_indices]
                logging.warning(
                    f"Found {invalid_tokens.sum()} invalid token IDs: values={invalid_values[:10].tolist()}, "
                    f"indices={invalid_indices[:10].tolist()}. Clamping to [0, {self.vocab_size-1}]"
                )
                # Fix invalid tokens
                self.token_ids = torch.clamp(self.token_ids, min=0, max=self.vocab_size - 1)

        self.token_len = len(self.token_ids)
        if self.token_len < self.seq_len + 1:
            raise ValueError(f"Token length ({self.token_len}) is too short for seq_len ({self.seq_len}).")

        logging.info(
            "TinyStoriesDataset initialized: token_len=%d, seq_len=%d, device=%s, streaming=%s, vocab_size=%d",
            self.token_len, self.seq_len, self.device, self.is_streaming, self.vocab_size
        )

    def __len__(self) -> int:
        """Return the number of samples in non-streaming mode.
        
        Raises:
            NotImplementedError: If in streaming mode.
        """
        if self.is_streaming:
            raise NotImplementedError("Length is not defined for streaming dataset.")
        return self.token_len - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample (input and target) by index."""
        if idx + self.seq_len + 1 > self.token_len:
            raise IndexError(
                f"Index {idx} + seq_len {self.seq_len} exceeds token_ids length {self.token_len}"
            )
        x = self.token_ids[idx:idx + self.seq_len].to(self.device, non_blocking=True)
        y = self.token_ids[idx + 1:idx + self.seq_len + 1].to(self.device, non_blocking=True)
        
        # Validate tokens
        if (x < 0).any() or (x >= self.vocab_size).any():
            logging.error(f"Invalid input tokens at index {idx}: x_min={x.min()}, x_max={x.max()}")
            raise ValueError(f"Invalid input tokens: x={x}")
        if (y < 0).any() or (y >= self.vocab_size).any():
            logging.error(f"Invalid target tokens at index {idx}: y_min={y.min()}, y_max={y.max()}")
            raise ValueError(f"Invalid target tokens: y={y}")
        
        # Debug logging
        logging.debug(f"Index {idx}: x_min={x.min().item()}, x_max={x.max().item()}, "
                      f"y_min={y.min().item()}, y_max={y.max().item()}")
        
        return x, y

    def __iter__(self):
        """Iterate over samples in streaming or non-streaming mode."""
        if not self.is_streaming:
            for idx in range(len(self)):
                yield self.__getitem__(idx)
        else:
            while True:
                idxs = torch.randint(
                    0, self.token_len - self.seq_len - 1,
                    (self.prefetch_size,), device='cpu'
                )
                for idx in idxs:
                    yield self.__getitem__(idx)

class DataLoaderFactory:
    """Factory class to create DataLoaders for training and validation."""
    
    def __init__(
        self,
        model_args: Llama4TextConfig,
        cfg: TrainingConfig,
        train_token_file: str = 'tokenized-train-samples_vocab-10k.pt',
        valid_token_file: str = 'tokenized-valid-samples_vocab-10k.pt',
        tokenizer_file: str = 'bpe-tokenizer_tinystories.json',
        pad_token: str = '</s>'
    ):
        """Initialize the DataLoaderFactory."""
        self.model_args = model_args
        self.cfg = cfg
        self.train_token_file = train_token_file
        self.valid_token_file = valid_token_file
        
        try:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, pad_token=pad_token)
            # Validate tokenizer vocab size
            if self.tokenizer.vocab_size != model_args.vocab_size:
                logging.warning(
                    f"Tokenizer vocab_size ({self.tokenizer.vocab_size}) does not match "
                    f"model vocab_size ({model_args.vocab_size}). This may cause issues."
                )
        except Exception as e:
            logging.error("Failed to load tokenizer from %s: %s", tokenizer_file, e)
            raise
        
        logging.info("DataLoaderFactory initialized with tokenizer vocab_size=%d", self.tokenizer.vocab_size)

    def create_train_loader(self) -> DataLoader:
        """Create the training DataLoader."""
        dataset = TinyStoriesDataset(
            token_file_path=self.train_token_file,
            seq_len=self.cfg.seq_len,
            is_streaming=True,
            device='cuda',
            prefetch_size=self.cfg.batch_size * 2,
            vocab_size=self.model_args.vocab_size
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            pin_memory=False,
            drop_last=True,
            num_workers=0
        )
        
        logging.info("Training DataLoader created: batch_size=%d, seq_len=%d",
                     self.cfg.batch_size, self.cfg.seq_len)
        return loader

    def create_valid_loader(self) -> DataLoader:
        """Create the validation DataLoader."""
        try:
            valid_ids = torch.load(self.valid_token_file, map_location='cpu')
        except Exception as e:
            logging.error("Failed to load validation token file %s: %s", self.valid_token_file, e)
            raise
        
        dataset = TinyStoriesDataset(
            token_ids=valid_ids,
            seq_len=self.cfg.seq_len,
            is_streaming=False,
            device='cuda',
            vocab_size=self.model_args.vocab_size
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            pin_memory=False,
            num_workers=0
        )
        
        logging.info("Validation DataLoader created: batch_size=%d, seq_len=%d",
                     self.cfg.batch_size, self.cfg.seq_len)
        return loader

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create both training and validation DataLoaders."""
        train_loader = self.create_train_loader()
        valid_loader = self.create_valid_loader()
        return train_loader, valid_loader