import torch

from raving_fader.config import settings, BaseConfig
from raving_fader.models import CRAVE
from raving_fader.pipelines.pipeline import Pipeline
from raving_fader.datasets.data_loaders import rave_data_loaders
from raving_fader.datasets.attr_dataset import get_dataset_attr
import os
import numpy as np


class CRAVEPipeline(Pipeline):
    def __init__(self, config: BaseConfig):
        super().__init__(config=config)

        self.name = self.train_config.name
        self.model_config = config.rave
        self.latent_length = self.data_config.n_signal // (self.model_config.data_size * np.prod(self.model_config.ratios))
        self.set_data_loaders_attr(latent_length=self.latent_length)
        self.set_model()

    def set_data_loaders(self):
        min_max_path = os.path.join(self.data_config.preprocessed, "min_max.pth")
        if os.path.exists(min_max_path):
            min_max_features = torch.load(min_max_path)
        else:    
            min_max_features = None

        self.dataset = get_dataset_attr(
            preprocessed=self.data_config.preprocessed,
            wav=self.data_config.wav,
            sr=self.data_config.sr,
            descriptors=self.data_config.descriptors,
            n_signal=self.data_config.n_signal,
            latent_length=self.latent_length,
            min_max_features=min_max_features)
        torch.save(self.dataset.min_max_features, min_max_path)

        self.train_set, self.val_set = rave_data_loaders(
            batch_size=self.train_config.batch,
            dataset=self.dataset,
            num_workers=settings.NUM_WORKERS)
        
        self.min_max_features = min_max_features

    def set_model(self):
        self.model = CRAVE(
            **dict(self.model_config),
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

        batch = x, features
        self.model.validation_step(batch)

        self.model.train(
            train_loader=self.train_set,
            val_loader=self.val_set,
            max_steps=self.train_config.max_steps,
            models_dir=self.train_config.models_dir,
            model_filename=self.name,
            display_step=100,
            ckpt=ckpt,
        )
