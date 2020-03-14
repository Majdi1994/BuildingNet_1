import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
device = torch.device('cpu')
import random
# discriminator network
def dicriminator():
    discriminator = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)
    return discriminator
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class CNN_1D(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_1D, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=11, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            Flatten(),
            nn.Linear(256, self.hidden_dim))
        self.Classifier= nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 11))
    def forward(self, src):
        features = self.encoder(src)
        predictions = self.Classifier(features)
        return predictions, features
