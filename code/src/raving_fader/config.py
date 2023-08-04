import os
import time
# import GPUtil as gpu
from typing import Optional, List, Union
from pydantic import BaseSettings, Field, BaseModel, validator


class DataConfig(BaseModel):
    sr: int = 48000
    n_signal: int = 65536
    preprocessed: str = None
    wav: str = None
    descriptors: list = []
    data_name: str = None
    r_samples: float = None
    nb_bins: int = 16


class RaveConfig(BaseModel):
    data_size: int = 16
    capacity: int = 64
    latent_size: int = 128
    ratios: List[int] = [4, 4, 4, 2]
    bias: bool = True

    loud_stride: int = 1

    use_noise: bool = True
    noise_ratios: List[int] = [4, 4, 4]
    noise_bands: int = 5

    d_capacity: int = 16
    d_multiplier: int = 4
    d_n_layers: int = 4

    warmup: int = 1000000
    mode: str = "hinge"
    no_latency: bool = False
    min_kl: float = 1e-1
    max_kl: float = 1e-1
    cropped_latent_size: int = 0
    feature_match: bool = True

    # latent_length: int = 32


class FaderConfig(BaseModel):
    initialize: int = 0
    num_lat_dis_layers: int = 2
    rave_ckpt: str = None
    clip_grad_norm: int = 5


class TrainConfig(BaseModel):
    name: str = None
    models_dir: str = None
    ckpt: str = None
    batch: int = 8
    max_steps: int = 3000000
    beta_inf: float = 0.1
    beta_delay: int = 30000
    lambda_inf: float = 1
    lambda_delay: int = 100000
    n_lat_dis: int = 1
    display_step: int = 1000
    val_check: int = None
    rave_mode: bool = False

    class Config:
        validate_assignment = True

    @validator('name')
    def set_name(cls, name):
        return name or f"rave_{time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime())}"


class BaseConfig(BaseModel):
    data: DataConfig = DataConfig()
    rave: RaveConfig = None
    fader: FaderConfig = None
    train: TrainConfig = TrainConfig()


class Settings(BaseSettings):
    MODELS_DIR: Optional[str] = Field(env="MODELS_DIR")
    DATA_DIR: Optional[str] = Field(env="DATA_DIR")
    CONFIG_DIR: Optional[str] = Field(env="CONFIG_DIR")
    # CUDA_VISIBLE_DEVICES: Optional[int] = Field(env="CUDA_VISIBLE_DEVICES")
    try:
        CUDA = gpu.getAvailable(maxMemory=.5)
    except Exception as e:
        print(e)
        CUDA = []
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS: int = 1  # Field(..., env="NUM_WORKERS")
    CUDA_VISIBLE_DEVICES: Optional[int] = Field(env="CUDA_VISIBLE")

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings(_env_file=os.environ.get("ENV_FILE") or ".env",
                    _env_file_encoding="utf-8")
print(settings.CUDA_VISIBLE_DEVICES)
print(settings.NUM_WORKERS)
