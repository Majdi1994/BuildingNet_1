import torch
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
# load your data 
# data_dir='./data/train_test_dataset_10s_11_classes.pt' 
data_dir='/content/drive/My Drive/data/train_test_dataset_10s_11_classes.pt'
data = torch.load(data_dir)
x,y=data['train_data'],data['train_labels']
print('Original dataset shape %s' % Counter(y.numpy()))
#appllying smote 
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(x, y)
print('Resampled dataset shape %s' % Counter(y_res))
train_data = torch.from_numpy(X_res).float()
train_labels = torch.from_numpy(y_res).long()
data['train_data'],data['train_labels'] =train_data,train_labels
torch.save(data, 'train_test_dataset_1024_smote_os.pt')
