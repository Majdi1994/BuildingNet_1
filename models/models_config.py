import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda')
def get_model_config(model_key,input_dim=14,hid_dim=32,n_layers=3,drop=0.2, bid=True):
    model_configs = {
        "CNN": {
            "model_name": "CNN",
            "input_dim": input_dim,
            "CLIP": 1,
            "permute": False,
            "out_dim": 32,
            "fc_drop": 0.5}}
    return model_configs[model_key]
