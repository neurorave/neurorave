import os
import torch
from abc import abstractmethod
from raving_fader.config import settings, BaseConfig
from raving_fader.datasets.rave_dataset import get_dataset
from raving_fader.datasets.data_loaders import rave_data_loaders
from raving_fader.datasets.attr_dataset import get_dataset_attr


class Pipeline:
    def __init__(self, config: BaseConfig):
        self.data_config = config.data
        self.train_config = config.train

        self.dataset = None
        self.train_set, self.val_set = None, None

        self.model = None

        # Check if a gpu is available and initialize device
        # if settings.CUDA_VISIBLE_DEVICES:
        #     use_gpu = int(int(settings.CUDA_VISIBLE_DEVICES) >= 0)
        # elif len(settings.CUDA):
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.CUDA[0])
        #     use_gpu = 1
        # elif torch.cuda.is_available():
        #     print("Cuda is available but no fully free GPU found.")
        #     print("Training may be slower due to concurrent processes.")
        #     use_gpu = 1
        # else:
        #     print("No GPU found.")
        #     use_gpu = 0

        # self.use_gpu = use_gpu
        # self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> Device : {self.device}")

    def set_data_loaders(self):
        self.dataset = get_dataset(
            preprocessed=self.data_config.preprocessed,
            wav=self.data_config.wav,
            sr=self.data_config.sr,
            n_signal=self.data_config.n_signal
        )
        self.train_set, self.val_set = rave_data_loaders(
            batch_size=self.train_config.batch,
            dataset=self.dataset,
            num_workers=settings.NUM_WORKERS
        )

    def set_data_loaders_attr(self, latent_length):
        features_path = os.path.join(self.data_config.preprocessed, "features.pth")
        try:
            allfeatures = torch.load(features_path)
        except:
            print('No features file Available')
            allfeatures = None

        self.dataset = get_dataset_attr(
            preprocessed=self.data_config.preprocessed,
            wav=self.data_config.wav,
            sr=self.data_config.sr,
            descriptors=self.data_config.descriptors,
            n_signal=self.data_config.n_signal,
            nb_bins=self.data_config.nb_bins,
            latent_length=latent_length,
            r_samples=self.data_config.r_samples,
            allfeatures=allfeatures)
        torch.save(self.dataset.allfeatures, features_path)

        self.train_set, self.val_set = rave_data_loaders(
            batch_size=self.train_config.batch,
            dataset=self.dataset,
            num_workers=settings.NUM_WORKERS)

    @abstractmethod
    def set_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass
