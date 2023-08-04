import torch

from raving_fader.config import BaseConfig
from raving_fader.models import RAVE
from raving_fader.pipelines.pipeline import Pipeline


class RAVEPipeline(Pipeline):
    def __init__(self, config: BaseConfig):
        super().__init__(config=config)

        self.model_config = config.rave
        self.name = self.train_config.name
        self.set_data_loaders()
        # initialize the RAVE model instance
        self.set_model()

    def set_model(self):
        self.model = RAVE(
            **dict(self.model_config),
            sr=self.data_config.sr,
            device=self.device
        ).to(self.device)

    def train(self, ckpt=None, it_ckpt=None):
        # this fake validation step is used to initialize the model
        # state_dict to allow to resume from a given checkpoint
        x = torch.zeros(self.train_config.batch, self.data_config.n_signal).to(self.device)
        self.model.validation_step(x)

        self.model.train(
            train_loader=self.train_set,
            val_loader=self.val_set,
            max_steps=self.train_config.max_steps,
            models_dir=self.train_config.models_dir,
            model_filename=self.name,
            display_step=100,
            ckpt=ckpt,
            it_ckpt=it_ckpt,
            rave_mode=self.train_config.rave_mode
        )
