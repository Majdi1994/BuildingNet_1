import torch
import matplotlib.pyplot as plt
device = torch.device('cpu')

def train(model, train_dl, optimizer, criterion,config):
    model.train()
    epoch_loss = 0
    epoch_score = 0
    total_pred= []
    total_labels=[]
    
    for inputs, labels in train_dl:
        src = inputs.unsqueeze(1).to(device)
       # print(src.shape)
        if config['permute']:
          src =src.permute(0,2,1)
        labels = labels.to(device)
        optimizer.zero_grad()
      #  print(src.shape)
        pred, feat = model(src)
        loss = criterion(pred, labels)
        #loss and score
        _, predicted_labels = torch.max(pred, 1)
        #loss and score
        total_pred.append(predicted_labels)
        total_labels.append(labels)
        loss.backward()
        if (config['model_name']=='LSTM'):
            clip=config['CLIP']
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # only for LSTM models
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(train_dl), total_pred, total_labels

def evaluate(model, test_dl, criterion, config):
    model.eval()
    total_pred=[];total_labels=[]
    epoch_loss = 0
    epoch_score = 0
    predicted_rul = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_dl:
            src = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)
            pred, feat = model(src)
            # loss and score
            loss = criterion(pred, labels)
            _, predicted_labels = torch.max(pred, 1)
            total_pred.append(predicted_labels)
            total_labels.append(labels)
            if (config['model_name'] == 'LSTM'):
                clip = config['CLIP']
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # only for LSTM models
            epoch_loss += loss
    model.train()
    return epoch_loss / len(test_dl),total_pred, total_labels