import signal
import queue
import time
import os
import glob
import sys

import torch

from raving_fader.realtime.run.audio import Audio
from raving_fader.realtime.run.network import Network
from raving_fader.realtime.run.realtimeFRAVE import RaspFRAVE

torch.set_grad_enabled(False)


def get_files_in_folder(dir_path, extension):
    res = []
    for file in os.listdir(dir_path):
        if file.endswith(extension):
            res.append(file)
    return res


class RaspiRave(object):
    def __init__(self):
        super(RaspiRave, self).__init__()
        # State
        self.descriptors = [
            "centroid", "rms", "bandwidth", "sharpness", "booming"
        ]

        self.blocksize_model = 8192

        self.sr = 44100
        self.audio_q = queue.Queue(10)
        self.playing = False

        self.attr_mod = {}
        self.volume = 1.

    def get_attr_mod(self):
        for state, value in self.state.items():
            if 'lfo' in state:
                self.attr_mod[state] = value

    def load_model(self, modelpth, audiofile):
        model = RaspFRAVE(modelpth, audiofile, self.blocksize_model, self.sr)
        self.blocksize = model.get_buffer().shape[0]
        return model

    def start_playing(self, model):
        self.audio_thread = Audio(model, self.sr, self.blocksize, self.audio_q,
                                  self.attr_mod, self.volume, self.playing)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def init_network(self):
        self.states = {
            "lfo_speed": 1.0,
            "lfo_amplitude": 0,
            "lfo_bias": 0.0,
            "lfo_waveform": "sine"
        }

        # self.modelspath = "/home/pi/raving-fader/models/FRAVE_DK"
        # self.audiopath = "/home/pi/raving-fader/audio_samples"

        self.modelspath = "../models/"
        self.audiopath = "../audio"

        self.available_models = get_files_in_folder(self.modelspath, ".ts")
        self.audio_samples_categories = next(os.walk(self.audiopath))[1]
        self.available_audio_samples = {}
        for c in self.audio_samples_categories:
            self.available_audio_samples[c] = get_files_in_folder(
                os.path.join(self.audiopath, c), ".wav")

        first_audio_sample = os.path.join(
            self.audio_samples_categories[0],
            self.available_audio_samples[self.audio_samples_categories[0]][0])

        self.state = {
            "model": self.available_models[0],
            "audio_sample": first_audio_sample,
            "output_volume": 1.0,
            "play": True
        }

        for descriptor in self.descriptors:
            for state, value in self.states.items():
                self.state[f"{descriptor}_{state}"] = value

        # self.get_attr_mod()
        # Network thread
        signal.signal(signal.SIGINT, self.signal_handler)
        self.rx = queue.Queue()
        self.network_thread = None
        self.network_thread = Network(
            self.rx, {
                "models": self.available_models,
                "audio_samples": self.available_audio_samples
            })
        self.network_thread.daemon = True

    def signal_handler(self, signum, frame):
        # Needed to close the other thread
        self.rx.put(None)
        exit(1)

    def launch(self):
        self.init_network()
        self.network_thread.start()

        modelpth = os.path.join(self.modelspath, self.state["model"])
        audiofile = os.path.join(self.audiopath, self.state["audio_sample"])

        model = self.load_model(modelpth, audiofile)

        while True:
            if self.rx.empty():
                pass
            tmp_buffer = []
            while not self.rx.empty():
                tmp_buffer.append(self.rx.get())
            for msg in tmp_buffer:
                print(f"{msg['type']} changed from {self.state[msg['type']]}",
                      end='')
                self.state[msg["type"]] = msg["state"]
                print(f" to {self.state[msg['type']]}")

                if msg["type"] == "play":
                    if (not self.playing) and msg["state"]:
                        self.playing = True
                        self.start_playing(model)

                    elif not msg["state"]:
                        self.audio_thread.playing = False
                        self.playing = False

                elif msg["type"] == "model" or msg["type"] == "audio_sample":
                    print('Loading Model')
                    self.playing = False
                    try:
                        self.audio_thread.playing = False
                    except:
                        pass

                    audio_sample = glob.glob(self.audiopath + "/**/" +
                                             self.state["audio_sample"],
                                             recursive=True)[0]

                    model = self.load_model(
                        os.path.join(self.modelspath, self.state["model"]),
                        audio_sample)
                    print('Model loaded')
                    self.playing = True
                    self.start_playing(model)

                elif msg["type"] == "output_volume":
                    self.volume = -0.01 + int(
                        self.state["output_volume"]) / 100
                    self.audio_thread.volume = self.volume

                self.get_attr_mod()

            time.sleep(0.01)


if __name__ == "__main__":
    try:
        app = RaspiRave()
        app.launch()
    except KeyboardInterrupt:
        print('Interruption')
        sys.exit()