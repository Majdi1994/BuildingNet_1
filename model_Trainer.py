import torch.nn as nn
from models.models import *
from models.models_config import get_model_config
from models.pre_train_test_split import trainer
import torch
from torch.utils.data import DataLoader
from utils import *
device = torch.device('cpu')
# load data
my_data= torch.load("/content/drive/My Drive/data/train_test_dataset_1024_smote_os.pt")
train_dl = DataLoader(MyDataset(my_data['train_data'], my_data['train_labels']), batch_size=256, shuffle=True, drop_last=True)
test_dl = DataLoader(MyDataset(my_data['test_data'], my_data['test_labels']), batch_size=10, shuffle=False, drop_last=False)
model = CNN_1D(1,256,0.5).to(device)
params = {'pretrain_epoch': 1000,'lr': 1e-3}
# load model
config = get_model_config('CNN')
# load data
trained_model=trainer(model, train_dl, test_dl,'SHM_C', config, params) 
