cLearning expressive control on RAVE for deep audio synthesis using Fader Networks 
=================================================================================

This project aims to implement on constrained hardware platforms 
a light-weight deep generative model generating audio in real-time 
and capable of expressive and understandable control to promote 
creative musical applications. We adapt the Fader Networks approach 
using continuous audio descriptors as control attributes over the RAVE 
generative process.

## Install :

- Clone the github repository :
```bash
$ git clone https://forge-2.ircam.fr/acids/team/collaboration/raving-fader.git
```
- Create a virtual environment with Python 3.9
- Activate the environment and install the dependencies with :
```bash
(myenv)$ pip install -r requirements.txt
```
- (Optional) Install the additional development dependencies to check the compliance to PEP8 with :
```bash
(myenv)$ pip install -r requirements-dev.txt
```

NB : If you encounter an error on the server while installing the requirements due to the package `sndfile`
try the following command :
```bash
(myenv)$ conda install -c conda-forge libsndfile
```
If CUDA kernel error :
```bash
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Declare the environment variables :
- Create a `.env` file at the root directory of the project
- Copy and paste the content of the `.env.template` file in it
- Replace the environment variables values with yours :
```bash
DATA_DIR=/data             # Absolute path to the dataset directory which contains an audio directory with the .wav files
MODELS_DIR=/models         # Absolute path to the models directory to store checkpoints and training configurations 
CONFIG_DIR=/config         # Absolute path to the configuration directory with pre-filled train config yaml files
CUDA_VISIBLE_DEVICES=-1    # ID of the GPU to use, if -1 use CPU
NUM_WORKERS=2              # Number of workers for the dataloader
```

These variables are processed by the `./src/raving_fader/config.py` file which instantiate the Settings 
class so that you can use these directly in your code. For instance :
```python
from raving_fader.config import settings

...
data_dir = settings.DATA_DIR
models_dir = settings.MODELS_DIR
...
```


## Project Structure :

```bash 
raving-fader
├── data    # directory to store the data in local /!\ DO NOT COMMIT /!\
├── docs    # to build Sphinx documentation based on modules docstrings and for github pages content
├── models  # directory to store the models checkpoints in local /!\ DO NOT COMMIT /!\
├── notebooks   # jupyter notebooks for data exploration and models analysis
├── README.md
├── requirements.txt   # python project dependencies with versions
├── requirements.txt   # python project development dependencies with versions
├── scripts   # scripts to executes pipelines (data preparation, training, evaluation) 
├── setup.py  # to create the python package library
├── src
│   └── raving_fader  # main package
│       ├── config.py      # global settings based on environment variables
│       ├── datasets       # data preprocessing and dataloader functions
│       │   └── __init__.py
│       ├── helpers        # global utility functions
│       │   └── __init__.py
│       ├── __init__.py
│       ├── models         # models architecture defined as class objects
│       │   └── __init__.py
│       └── pipelines      # pipelines for data preparation, training and evaluation for a given model
│           └── __init__.py
└── tests                        # tests package with unit tests
    ├── conftest.py
    └── __init__.py

```


## Run on server

First create `data` and `models` directories

1. Go to raving_fader directory and open a screen session
```bash
$ screen -S "name"  # to create a new screen

$ screen -r "name"  # to rattach if screen already exists
```
NB : To exit a screen but not delete it do : ctrl + A and then ctrl + D (if you do just ctrl + D)

2. activate python virtual environment. If you use miniconda just do:
```bash
$ source ../miniconda/bin/activate  # change absolute path if needed
```
NB : do not forget to install the dependencies the first time with :
```bash
(myenv)$ pip install -r requirements.txt

```

3. export environment variables:
```bash
$ export PYTHONPATH=$PYTHONPATH:${PWD}/src
$ export DATA_DIR=${PWD}/data      # CHANGE if needed
$ export MODELS_DIR=${PWD}/models  # CHANGE if needed
$ export CONFIG_DIR=${PWD}/config  # CHANGE if needed
$ export NUM_WORKERS=8             # CHANGE if needed
```

4. specify the GPU to use:
check the available GPU with `$ nvidia-smi`
```bash
$ export CUDA_VISIBLE_DEVICES=1  # replace the value with the GPU id you want to use
```

5. Launch training: 

The available models are :

- rave : The RAVE model without Pytorch Lightning. Use : 
  ```
  (myenv) $ python -m raving_fader train rave
  ```
- ravelight : The RAVE model with Pytorch Lightning implementation. Use : 
  ```
  (myenv) $ python -m raving_fader train ravelight
  ```


To see training options use : `(myenv) $ python -m raving_fader train --help`

Output:
```bash
Usage: python -m raving_fader train [OPTIONS] MODEL

  Available models :     - rave     - ravelight

Options:
  --data_dir TEXT         Absolute path to data audio directory which contains
                          an audio directory with the .wav files, default is
                          the $DATA_DIR environment variable
  --models_dir TEXT       Absolute path to the models directory to store
                          checkpoints and training configurations, default is
                          the $MODELS_DIR environment variable
  --config_dir TEXT       Absolute path to the configuration directory with
                          pre-filled train config yaml files, default is the
                          $MODELS_DIR environment variable
  -f, --config_file TEXT  Name of the model's yaml configuration file to use
                          for train, default is rave_config.yaml
  -n, --name TEXT         Name of the model can also be specify in the yaml
                          configuration fileif not specified (None) the name
                          would be set to 'rave_{timestamp}'
  --ckpt TEXT             Filepath to model checkpoint to resume training
                          from, default in yaml config file
  --sr TEXT               Audio data sampling rate, default in yaml config
                          file
  --batch TEXT            batch size, default in yaml config file, default in
                          yaml config file
  --max_steps TEXT        number of total training steps, default in yaml
                          config file
  --warmup TEXT           number of training steps for the first stage
                          representation learning, default in yaml config file
  --help                  Show this message and exit.
```

