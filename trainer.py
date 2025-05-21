import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import math 
class TrainingConfig:
    def __init__(
        self,
        batch_size=16,
        seq_len=128,
        epochs=4,
        steps_per_epoch=1000,
        report_interval=100,
        grad_clip_norm=1.0,
        learning_rate=1e-4,
        warmup_steps=100,
        max_lr=1e-3,
        min_lr=1e-5,
        log_interval=10,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.report_interval = report_interval
        self.grad_clip_norm = grad_clip_norm
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.log_interval = log_interval

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        device,
        cfg: TrainingConfig
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.tokenizer = tokenizer
        self.device = device
        self.config = cfg

        if self.tokenizer.pad_token_id is None:
            logging.warning("pad_token_id is None, setting to default 0")
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

        logging.info("Trainer initialized on %s", self.device)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        use_cuda = self.device.startswith("cuda")
        loader = tqdm(self.train_loader,
                      desc=f"Epoch {epoch} [Train]",
                      total=self.config.steps_per_epoch,
                      leave=False)
        for step, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = self.model(inputs)
                logits = outputs.logits.view(-1, outputs.logits.size(-1))
                targets_flat = targets.view(-1)
                loss = F.cross_entropy(
                    logits,
                    targets_flat,
                    ignore_index=self.tokenizer.pad_token_id
                )

            if use_cuda:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            if use_cuda:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()

            loader.set_postfix(loss=f"{loss.item():.4f}")
            if step % self.config.log_interval == 0:
                logging.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
            if step >= self.config.steps_per_epoch:
                break

        avg_loss = total_loss / self.config.steps_per_epoch
        logging.info("Epoch %d done. Avg Loss: %.4f", epoch, avg_loss)
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        valid_len = len(self.valid_loader)
        loader = tqdm(self.valid_loader,
                      desc="Validation",
                      total=valid_len,
                      leave=False)
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                logits = outputs.logits.view(-1, outputs.logits.size(-1))
                targets_flat = targets.view(-1)
                loss = F.cross_entropy(
                    logits,
                    targets_flat,
                    ignore_index=self.tokenizer.pad_token_id
                )
                total_loss += loss.item()
                loader.set_postfix(val_loss=f"{loss.item():.4f}")

        avg_loss = total_loss / valid_len
        logging.info("Validation Avg Loss: %.4f", avg_loss)
        return avg_loss

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        top_k: int = 50,
        temperature: float = 1.0
    ) -> str:
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.seq_len
            ).input_ids.to(self.device)
            generated = inputs.clone()
            batch_size = generated.size(0)
            for _ in range(max_new_tokens):
                outputs = self.model(generated)
                logits = outputs.logits[:, -1, :]  # [batch, vocab]
                logits = logits / temperature
                # top-k sampling
                topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)  # [batch, top_k]
                probs = F.softmax(topk_vals, dim=-1)  # [batch, top_k]
                # sample index within top_k
                sample_idx = torch.multinomial(probs, num_samples=1)  # [batch, 1]
                # map to vocab index
                next_token = topk_idx.gather(-1, sample_idx)  # [batch,1]
                # append
                generated = torch.cat([generated, next_token], dim=1)  # [batch, seq+1]
                # stop if any EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            return self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

    def train(self, prompts=None):
        prompts = prompts or ["Hello world!", "The meaning of life is", "Once upon a time"]
        for epoch in range(1, self.config.epochs + 1):
            logging.info(f"=== Epoch {epoch}/{self.config.epochs} ===")
            train_loss = self.train_epoch(epoch)
            logging.info(f"Epoch {epoch} complete. Train Loss: {train_loss:.4f}")
            for p in prompts:
                sample = self.generate_text(p, max_new_tokens=self.config.seq_len)
                logging.info(f"Sample at epoch {epoch}: {sample}")
            if epoch==4:
                val_loss = self.evaluate()
                logging.info(f"Epoch {epoch} complete. Val Loss: {val_loss:.4f}")
        return
