import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from trainer import TrainingConfig, Trainer
from model import Llama4TextConfig, Llama4TextModel,Llama4ForCausalLM
from dataloader import DataLoaderFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)

def main():
    """Main function to set up and run the training process."""
    # Define model configuration
    model_args = Llama4TextConfig(
        vocab_size=10000,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=12,
        num_attention_heads=32,
        num_key_value_heads=8,
        moe_layers=[6],
        num_local_experts=6,
        max_position_embeddings=128,
        use_flash_attention=False,
        attention_dropout=0.1,
        initializer_range=0.02,
    )

    # Training configuration
    cfg = TrainingConfig(
        batch_size=16,  # Reduced due to larger model size
        seq_len=128,
        epochs=4,
        steps_per_epoch=10000,
        report_interval=1000,
        grad_clip_norm=1.0,
        learning_rate=1e-4,
        warmup_steps=10,
        max_lr=1e-4,
        min_lr=1e-5,
        log_interval=1000
    )

    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Initialize model
    try:
        model = Llama4ForCausalLM(model_args).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info("Model initialized with %d parameters (~%.2fM)", total_params, total_params / 1e6)
    except Exception as e:
        logging.error("Failed to initialize model: %s", e)
        raise

    # Create data loaders with DataLoaderFactory
    try:
        dl_factory = DataLoaderFactory(
            model_args=model_args,
            cfg=cfg,
            train_token_file='tokenized-train-samples_vocab-10k.pt',
            valid_token_file='tokenized-valid-samples_vocab-10k.pt',
            tokenizer_file='bpe-tokenizer_tinystories.json',
            pad_token='</s>'
        )
        train_loader, valid_loader = dl_factory.create_data_loaders()
        tokenizer = dl_factory.tokenizer
    except Exception as e:
        logging.error("Failed to create data loaders: %s", e)
        raise

    logging.info("Data loaders created: %d validation batches, training in streaming mode",
                 len(valid_loader))

    # Initialize tokenizer fallback if needed
    if tokenizer is None:
        logging.warning("Using fallback tokenizer (GPT-2)")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logging.error("Failed to load fallback tokenizer: %s", e)
            raise

    # Initialize optimizer and scheduler
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs * cfg.steps_per_epoch,
            eta_min=cfg.min_lr
        )
        scaler = GradScaler() if device == "cuda" else None
    except Exception as e:
        logging.error("Failed to initialize optimizer or scheduler: %s", e)
        raise

    # Define prompts for text generation
    prompts = ["Hello world!", "The meaning of life is", "Once upon a time"]

    # Initialize trainer
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokenizer=tokenizer,
            device=device,
            cfg=cfg
        )
        logging.info("Trainer initialized")
    except Exception as e:
        logging.error("Failed to initialize trainer: %s", e)
        raise

    # Run training
    try:
        for epoch in range(1, cfg.epochs + 1):
            trainer.train(prompts=prompts)
            checkpoint_path = f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logging.error("Training failed: %s", e)
        raise

    # Save final model
    final_path = "model_final.pth"
    torch.save(model.state_dict(), final_path)
    logging.info(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()