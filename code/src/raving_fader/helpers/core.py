import yaml
import os
from glob import glob
import pickle


def save_pickle(data_dict, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath):
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
    return data


def load_config(yaml_filepath):
    with open(yaml_filepath, 'r') as s:
        try:
            parsed_yaml = yaml.safe_load(stream=s)
            config = parsed_yaml
        except yaml.YAMLError as e:
            print(e)
            config = {}
        finally:
            return config


def write_config(yaml_filepath, data):
    with open(yaml_filepath, 'w') as s:
        try:
            yaml.safe_dump(data, stream=s)
        except yaml.YAMLError as e:
            print(e)


def search_for_run(run_path):
    if run_path is None:
        return None

    if ".ckpt" in run_path:
        pass
    elif "checkpoints" in run_path:
        run_path = os.path.join(run_path, "*.ckpt")
        run_path = glob(run_path)
        run_path = list(filter(lambda e: "last" in e, run_path))[-1]
    elif "version" in run_path:
        run_path = os.path.join(run_path, "checkpoints", "*.ckpt")
        run_path = glob(run_path)
        run_path = list(filter(lambda e: "last" in e, run_path))[-1]
    else:
        run_path = glob(os.path.join(run_path, "*"))
        run_path.sort()
        if len(run_path):
            run_path = run_path[-1]
            run_path = os.path.join(run_path, "checkpoints", "*.ckpt")
            run_path = glob(run_path)
            run_path = list(filter(lambda e: "last" in e, run_path))[-1]
        else:
            run_path = None
    return run_path
