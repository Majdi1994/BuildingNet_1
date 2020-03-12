import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda')
#from models.models import lstm,VRNN_model,CNN,ResNet,resnet34,resnet18,TCN, seq2seq
def get_model_config(model_key,input_dim=14,hid_dim=32,n_layers=3,drop=0.2, bid=True):
    model_configs = {
        "CNN": {
            "model_name": "CNN",
            "input_dim": input_dim,
            "CLIP": 1,
            "permute": False,
            "out_dim": 32,
            "fc_drop": 0.5},
        "LSTM": {
            "model_name": 'LSTM',
            "input_dim": input_dim,
            "hid_dim": hid_dim,
            "out_dim": input_dim,
            "n_layers": 3,
            "permute": True,
            "bid": True,
            "drop": 0.5,
            "CLIP": 1,
            "save":False,
            "fc_drop": 0.5
        },
        "Transformer": {
            "model_name": 'Transformer',
            "input_dim": 14,
            "hid_dim": 200,  # the dimension of the feedforward network model in nn.TransformerEncoder
            "out_dim": 1,
            "n_layers": 2,
            "nhead" : 1,  # the number of heads in the multiheadattention models
            "dropout" : 0.2 , # the dropout value
            "permute": False,
            "bid": True,
            "drop": 0.5,
            "CLIP": 1,
            "save": False,
            "fc_drop": 0.5
        },
        "VRNN": {
            "model_name": 'VRNN',
            "input_dim": input_dim,
            "hid_dim": hid_dim,
            "z_dim": 32,
            "permute":False,
            "n_layers": n_layers,
            "drop": 0.2,
            "CLIP": 1,
            "fc_drop": 0.2},
        "RESNET": {
            "model_name": 'RESNET',
            "input_dim": input_dim,
            "num_classes": 1,
            "permute": True
            },
        "TCN":{"model_name": "TCN",
         "input_channels": input_dim,
         "n_classes": 1,
         "permute":True,
         "num_channels": [8] * 4,
         "kernel_size": 16,
         "drop": 0.2
               },
        "seq2seq":{
            "model_name": 'seq2seq',
            "input_dim": input_dim,
            "hid_dim": hid_dim,
            "out_dim": input_dim,
            "n_layers": 3,
            "permute": False,
            "bid": False,
            "drop": 0.5,
            "CLIP": 1,
            "fc_drop": 0.5}
    }
    return model_configs[model_key]