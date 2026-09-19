import math
import torch
import pytorch_lightning as pl

from src.models.gpt import GPT


class GPTLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        self.model = GPT(
            vocab_size=cfg.model.vocab_size,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers,
            d_ff=cfg.model.d_ff,
            max_seq_len=cfg.model.max_seq_len,
            dropout=cfg.model.dropout,
        )

    def forward(self, input_ids, seq_ids):
        return self.model(input_ids, seq_ids)

    def _shared_step(self, batch, stage: str):
        input_ids, seq_ids = batch

        # Ранняя диагностика: ловим несоответствие vocab_size и данных
        # до CUDA-ассерта (который всплывает асинхронно и сбивает стектрейс).
        if stage == "train" and self.global_step == 0:
            max_id = int(input_ids.max())
            assert max_id < self.cfg.model.vocab_size, (
                f"input_ids содержит токен {max_id}, но vocab_size="
                f"{self.cfg.model.vocab_size}. Увеличь vocab_size в конфиге "
                f"или пересоздай датасет под текущий токенизатор."
            )

        logits = self(input_ids, seq_ids)
        loss = self.model.compute_loss(logits, input_ids, seq_ids)
        perplexity = torch.exp(loss)

        self.log(f"{stage}/loss", loss, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/perplexity", perplexity, prog_bar=True,
                 on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

        warmup_steps = int(self.cfg.training.warmup_steps)

        # Lightning считает это из len(train_dataloader) * max_epochs
        # (с учётом accumulate_grad_batches). Безопасно на любом этапе.
        total_steps = max(
            int(getattr(self.trainer, "estimated_stepping_batches", 0)),
            warmup_steps + 1,
        )
        decay_steps = max(1, total_steps - warmup_steps)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(decay_steps)
            progress = min(progress, 1.0)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }

    def on_after_backward(self):
        # clip_grad_norm_ с max_norm=inf просто возвращает норму,
        # ничего не клиппит. Быстро, без питоновского цикла и .item().
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        self.log("train/grad_norm", total_norm,
                 on_step=True, on_epoch=False, prog_bar=False)
