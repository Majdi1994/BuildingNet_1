import torch
from torch import optim
import time
from utils import *
from torch.optim.lr_scheduler import StepLR
from models.train_eval import train, evaluate
import torch.nn as nn
from sklearn.metrics import classification_report
device = torch.device('cpu')
def trainer(model, train_dl, test_dl, data_id, config, params):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    target_names = ['Healthy','D1','D2','D3','D4','D5','D6','D7', 'D8','D9','D10']
    for epoch in range(params['pretrain_epoch']):
        start_time = time.time()
        train_loss, train_pred, train_labels = train(model, train_dl, optimizer, criterion, config)
        scheduler.step()
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # printing results
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        # Evaluate on the test set
        test_loss,_, _= evaluate(model, test_dl, criterion, config)
        print('=' * 89)
        print(f'\t  Performance on test set::: Loss: {test_loss:.3f} ')
        train_labels = torch.stack(train_labels).view(-1)
        train_pred = torch.stack(train_pred).view(-1)
        print(classification_report(train_labels, train_pred, target_names=target_names))
    # Evaluate on the test set
    test_loss, y_pred, y_true = evaluate(model, test_dl, criterion, config)
    y_true = torch.stack(y_true).view(-1)
    y_pred = torch.stack(y_pred).view(-1)
    print('=' * 50)
    print(f'\t  Performance on test set:{data_id}::: Loss: {test_loss:.3f} ')
    print('=' * 50)
    print(classification_report(y_true, y_pred, target_names=target_names))
    print('| End of Pre-training  |')
    print('=' * 50)
    return model
