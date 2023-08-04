import torch
import numpy as np

from raving_fader.config import BaseConfig
from raving_fader.pipelines.pipeline_rave import Pipeline
from raving_fader.models import FadeRAVE


class FadeRAVEPipeline(Pipeline):
    def __init__(self, config: BaseConfig,inst_loaders=True):
        super().__init__(config=config)

        self.name = self.train_config.name
        self.model_config = config.rave
        self.fader_config = config.fader
        self.latent_length = self.data_config.n_signal // (self.model_config.data_size * np.prod(self.model_config.ratios))
        if inst_loaders==True:
            self.set_data_loaders_attr(latent_length=self.latent_length)
        self.set_model()

    def set_model(self):
        self.model = FadeRAVE(
            **dict(self.model_config),
            **dict(self.fader_config),
            sr=self.data_config.sr,
            descriptors=self.data_config.descriptors,
            device=self.device
        ).to(self.device)

    def train(self, ckpt=None, it_ckpt=None):
        # this fake validation step is used to initialize the model
        # state_dict to allow to resume from a given checkpoint
        x = torch.zeros(self.train_config.batch,
                        self.data_config.n_signal).to(self.device)
        features = torch.zeros(self.train_config.batch,
                               len(self.data_config.descriptors),
                               self.latent_length).to(self.device)
        self.model.validation_step(x, features)

        self.model.train(
            train_loader=self.train_set,
            val_loader=self.val_set,
            model_filename=self.name,
            it_ckpt=it_ckpt,
            bin_values=self.dataset.bin_values,
            dataset=self.dataset,
            **dict(self.train_config)
        )
