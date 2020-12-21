import numpy as np
import math
import torch
import torch.nn as nn

from data_loader import *
from network import *
from config import*

network=create_model()

def load_net():
    model_dict=torch.load(config["model_path"])
    # print(model_dict.keys())
    network.load_state_dict(model_dict["net"])
    # print(network.state_dict().keys())
    conv1_weight=network.conv1.weight
    # print(conv1_weight.shape)
    # print(type(conv1_weight))
    conv1_weight_np=conv1_weight.detach().numpy()
    # print(conv1_weight_np)
    conv1_weight_np=np.squeeze(conv1_weight_np,axis=1)
    print(conv1_weight_np.shape)
    print(conv1_weight_np[1,:,:])

if __name__ == '__main__':
    load_net()
