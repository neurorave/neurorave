import os
import sys
sys.path.append('./src')
# print(sys.path)
from raving_fader.config import settings, RaveConfig, DataConfig, ModelConfig
from raving_fader.helpers.core import load_config
from raving_fader.datasets.attr_dataset import get_dataset_attr
from raving_fader.datasets.data_loaders import rave_data_loaders


config_dir = "./config/"
config_file = "rave_config.yaml"
data_dir = r"C:\Users\NILS\Documents\ATIAM\Stage\Work\simple_vae\data\dataset\NSYNTH\small_test\audio_test"


config = load_config(os.path.join(config_dir, config_file))

data_config = DataConfig(**config["data"])
model_config = ModelConfig(**config["model"])


name = model_config.name

data_config.wav =data_dir
data_config.preprocessed = os.path.join(data_dir, name, "tmp")


config = RaveConfig(data=data_config, model=model_config)


config.data.sr=16000
n_band = 16
ratio = 4 * 4 * 4 * 2
latent_length = int(config.data.n_signal/n_band/ratio)
print(config.data.descriptors)


dataset = get_dataset_attr(
            preprocessed=config.data.preprocessed,
            wav=config.data.wav,
            sr=config.data.sr,
            descriptors=config.data.descriptors,
            n_signal=config.data.n_signal,
            latent_length=latent_length
            )



train_loader,valid_loader = rave_data_loaders(4, dataset, num_workers=8)

print(next(iter(train_loader)))
# # print(dataset.__getitem__(0))
# import numpy as np
# print(latent_length)
# data,features = dataset.__getitem__(0)
# print(data.shape)
# for feature,val in features.items():
#     print(val.shape)
#     print(feature)
        
#     print(np.std(val))
#     print(np.mean(val))
#     print(np.max(val))
#     print(np.min(val))
    
