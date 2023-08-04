import os
import json
import torch
import librosa
import numpy as np
import torch.nn.functional
from torch.utils.data import Dataset


class AttrPrior(Dataset):
    def __init__(self,
                 feature_path,
                 ref_descriptors = None,
                 transform=None):
        super(AttrPrior, self).__init__()
        
        #self.allfeatures_reference = torch.load(os.path.join(eval_path, "FRAVE_multi_features.pth"))
        self.allfeatures_reference = torch.load(feature_path)
        
        if ref_descriptors is not None :
            self.ref_descriptors = ref_descriptors
        else :
            self.ref_descriptors = ["centroid","rms","bandwidth","sharpness","booming"]
        
        self.min_max_ref = {}
        # Compute min max for referenced descriptors
        for i, descr in enumerate(self.ref_descriptors) :
            self.min_max_ref[descr] = [np.min(self.allfeatures_reference[:,i,:]), 
                                       np.max(self.allfeatures_reference[:,i,:])]
        
    def normalize(self, array, min_max) :
        return -1 + 2 * ( (array - min_max[0]) / (min_max[1] - min_max[0]) )
    
    def denormalize(self, array, min_max) :
        return min_max[0] + ((array + 1) * (min_max[1] - min_max[0]) / 2)

    def __len__(self):
        return self.allfeatures_reference.shape[0]

    def __getitem__(self, index):
        # Normalize referenced descriptors 
        descriptors = np.zeros((len(self.ref_descriptors), self.allfeatures_reference.shape[-1]))

        for i, descr in enumerate(self.ref_descriptors) :
            descriptors[i,:] = self.normalize(self.allfeatures_reference[index,i,:], 
                                                  self.min_max_ref[descr])

        return descriptors

def get_dataset_attr(feature_path,
                 ref_descriptors = None):
    dataset = AttrPrior(feature_path,
                 ref_descriptors,
                 transform=None
        )
    return dataset