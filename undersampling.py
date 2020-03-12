import torch
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids
# orignal data 
# load your data 
data_dir='./data/train_test_dataset_10s_11_classes.pt'
data = torch.load(data_dir)
x,y=data['train_data'],data['train_labels']
print('Original dataset shape %s' % Counter(y.numpy()))
#appllying smote 
cc = ClusterCentroids(random_state=0)
X_res, y_res = cc.fit_resample(x, y)
print(sorted(Counter(y_resampled).items()))
train_data = torch.from_numpy(X_res).float()
train_labels = torch.from_numpy(y_res).long()
data['train_data'],data['train_labels'] = train_data,train_labels
torch.save(data, 'train_test_dataset_1024_smote_us.pt')