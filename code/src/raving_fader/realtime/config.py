import os 

from raving_fader.config import DataConfig, RaveConfig, FaderConfig, TrainConfig, BaseConfig
from raving_fader.helpers.core import load_config    
    
def create_config(config_path):   
    config = load_config(config_path)
    data_config = DataConfig(**config["data"])

    # >>> MODEL
    rave_config = RaveConfig(**config["rave"])
    fader_config = FaderConfig(rave=rave_config, **config["fader"])
    train_config = TrainConfig(**config["train"])
    config = BaseConfig(data=data_config, rave=rave_config, fader=fader_config, train=train_config)
    return config