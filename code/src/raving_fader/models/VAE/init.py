import os
import shutil
import random

import torch
import numpy as np


def setup_directories(config):
    # Handling directories
    dataset_name = config["data"]["data_location"].split("/")[-1]
    model_variants = [dataset_name, config["model"]["fader"], config["model"]["n_attr"], config["discriminator"]["lambda_lat_dis"]]
    #model_variants = [config["data"]["dataset"], config["model"]["fader"], config["model"]["n_attr"], config["discriminator"]["lambda_lat_dis"]]
    config["data"]["final_path"] = config["data"]["output_path"] + '/'
    for m in model_variants:
        config["data"]["final_path"] += str(m) + '_'
    config["data"]["final_path"] = config["data"]["final_path"][:-1] + '/'
    if os.path.exists(config["data"]["final_path"]):
        shutil.rmtree(config["data"]["final_path"])
    else:
        os.makedirs(config["data"]["final_path"])
    # Create all sub-folders
    config["data"]["model_path"] = config["data"]["final_path"] + 'models/'
    config["data"]["losses_path"] = config["data"]["final_path"] + 'losses/'
    config["data"]["figures_path"] = config["data"]["final_path"] + 'figures/'
    config["data"]["wav_results_path"] = config["data"]["final_path"] + 'wav/'
    config["data"]["dataset_features_path"] = config["data"][
        "final_path"] + 'dataset_features/'
    config["data"]["stats_path"] = config["data"]["final_path"] + 'stats/'
    for p in [
            config["data"]["model_path"], config["data"]["losses_path"],
            config["data"]["figures_path"], config["data"]["wav_results_path"],
            config["data"]["dataset_features_path"],
            config["data"]["stats_path"]
    ]:
        os.makedirs(p)
    data_variants = [
        dataset_name, config["data"]["representation"]
    ]
    #data_variants = [config["data"]["dataset"], config["data"]["representation"]]
    config["data"][
        "loaders_path"] = config["data"]["output_path"] + '/loaders_'
    for m in data_variants:
        config["data"]["loaders_path"] += str(m) + '_'
    config["data"]["loaders_path"] = config["data"]["loaders_path"][:-1]
    return config


def init(config):
    # Set seeds
    torch.manual_seed(config["train"]["seed"])
    np.random.seed(config["train"]["seed"])
    random.seed(config["train"]["seed"])
    # Enable CuDNN optimization
    if config["train"]["device"] != 'cpu':
        torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor
        ltype = torch.LongTensor
    # Handling cuda
    torch.cuda.set_device(config["train"]["device"])
    config["train"]["cuda"] = not config["train"][
        "device"] == 'cpu' and torch.cuda.is_available()
    config["train"]["device"] = torch.device(
        config["train"]["device"] if torch.cuda.is_available() else 'cpu')

    # Print info
    print(10 * '*******')
    print('* Lovely run info:')
    print('** Your great optimization will be on ' +
          str(config["train"]["device"]))
    print('** Your wonderful model is ' + str(config["model"]["model"]))
    print('** You are using the schwifty ' + str(config["data"]["dataset"]) +
          ' dataset')
    print('*** Using fader = ' + str(config["model"]["fader"]))
    print(10 * '*******')
    return setup_directories(config)
