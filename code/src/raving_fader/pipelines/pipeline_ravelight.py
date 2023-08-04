import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from raving_fader.models import RAVELight
from raving_fader.config import BaseConfig
from raving_fader.helpers.core import search_for_run
from raving_fader.pipelines.pipeline import Pipeline


class EMAModelCheckPoint(ModelCheckpoint):
    def __init__(self, model: torch.nn.Module, alpha=.999, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()
        self.model = model
        self.alpha = alpha

    def on_train_batch_end(self, *args, **kwargs):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.shadow:
                    self.shadow[n] *= self.alpha
                    self.shadow[n] += (1 - self.alpha) * p.data

    def on_validation_epoch_start(self, *args, **kwargs):
        self.swap()

    def on_validation_epoch_end(self, *args, **kwargs):
        self.swap()

    def swap(self):
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                tmp = p.data.clone()
                p.data.copy_(self.shadow[n])
                self.shadow[n] = tmp

    def save_checkpoint(self, *args, **kwargs):
        self.swap()
        super().save_checkpoint(*args, **kwargs)
        self.swap()


class RAVELightPipeline(Pipeline):
    def __init__(self, config: BaseConfig):
        super().__init__(config=config)

        self.model_config = config.rave
        self.name = self.train_config.name
        self.set_data_loaders()
        # initialize the RAVELight model instance
        self.set_model()

    def set_model(self):
        self.model = RAVELight(
            **dict(self.model_config),
            sr=self.data_config.sr,
        )

    def train(self, ckpt=None):
        x = torch.zeros(self.train_config.batch, 2**14)
        self.model.validation_step(x, 0)

        # CHECKPOINT CALLBACKS
        validation_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="validation",
            filename="best",
        )
        last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

        val_check = {}
        if len(self.train_set) >= 10000:
            val_check["val_check_interval"] = 10000
        else:
            nepoch = 10000 // len(self.train_set)
            val_check["check_val_every_n_epoch"] = nepoch

        trainer = pl.Trainer(
            logger=pl.loggers.TensorBoardLogger(os.path.join(
                "runs", self.name),
                                                name="rave"),
            gpus=self.use_gpu,
            callbacks=[validation_checkpoint, last_checkpoint],
            resume_from_checkpoint=search_for_run(self.train_config.ckpt),
            max_epochs=100000,
            max_steps=self.train_config.max_steps,
            **val_check,
        )
        trainer.fit(self.model, self.train_set, self.val_set)
