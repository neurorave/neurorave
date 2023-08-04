import numpy as np
import sounddevice as sd


sr_1 = 44100
sr_2 = 48000
t1 = np.linspace(0,2,2*sr_1)
t2 = np.linspace(0,2,2*sr_2 )
y1 = 0.1*np.cos(2*np.pi*440*t1)
y2 = 0.1*np.cos(2*np.pi*440*t2)
sd.play(y1)
# sd.play(y2)
sd.wait()