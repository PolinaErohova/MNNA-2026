import os
from dotenv import load_dotenv
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.lightning_module import GPTLightningModule
from src.data.wikitext_datamodule import WikiTextDataModule


def main():
    # 1. Переменные окружения
    load_dotenv("/content/MNNA-2026/.env")

    data_dir = os.getenv("DATA_DIR", "/content/drive/MyDrive/CommonCrawl")
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/content/drive/MyDrive/CommonCrawl/checkpoints")
    project_name = os.getenv("CLEARML_PROJECT_NAME", "MNNA-2026")

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Чекпоинты будут сохраняться в: {checkpoint_dir}")

    # 2. Конфиг
    cfg = OmegaConf.load("/content/MNNA-2026/configs/gpt_config.yaml")
    print("Конфиг:")
    print(OmegaConf.to_yaml(cfg))

    # 3. DataModule
    npz_path = os.path.join(data_dir, "wikitext_packed_batches.npz")
    dm = WikiTextDataModule(
        npz_path=npz_path,
        batch_size=cfg.training.batch_size,
        val_split=0.05,
    )

    # 4. Модель
    model = GPTLightningModule(cfg)

    # 5. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="gpt-{epoch:02d}-{val_perplexity:.2f}",
        monitor="val/perplexity",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 6. Логгер
    tb_logger = TensorBoardLogger(
        save_dir="/content/MNNA-2026/logs",
        name="gpt",
    )

    # 7. Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=cfg.training.max_grad_norm,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=50,
        val_check_interval=0.5,
        precision="16-mixed",
    )

    # 8. Resume из последнего чекпоинта, если есть
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    resume_path = last_ckpt if os.path.exists(last_ckpt) else None
    if resume_path:
        print(f"Продолжаем обучение с чекпоинта: {resume_path}")

    # 9. Запуск
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path)

    print("\nЛучший чекпоинт:", checkpoint_callback.best_model_path)
    print("Лучшая perplexity:", checkpoint_callback.best_model_score)


if __name__ == "__main__":
    main()
