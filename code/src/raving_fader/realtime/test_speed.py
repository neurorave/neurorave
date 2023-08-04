import torch
import time
import numpy as np


model_name="FRAVE_DK"
model_path = "../models/"+model_name + "/"


model = torch.jit.load(model_path+"rave_32.ts")



buff_size = 2048


for buff_size in [2**i for i in range(8,14)]:
    time_audio = buff_size/model.sr

    x = torch.zeros((1,1,buff_size))
    
    times = []
    for i in range(100):
        start = time.time()
        z = model.encode(x)
        attr= torch.zeros(1,5,z.shape[-1])
        
        zc = torch.cat((z,attr),dim=1)
        _ = model.decode(zc).squeeze().detach().numpy()
        times.append(time.time()-start)


    time_inf = np.mean(times)

    print(buff_size)
    print(round(1000*time_audio,1))
    print(round(100*time_inf/time_audio,2)," %")
    print('#'*40)
    