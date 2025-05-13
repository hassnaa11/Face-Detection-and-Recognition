import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 

#generating synthetic data
N_classes = 3
N_per_class=100
labels = np.concatenate([[i]*N_per_class for i in range(N_classes)])
preds = np.stack([np.random.uniform(0,1,N_per_class*N_classes) for _ in range(N_classes)]).T
preds /= preds.sum(1,keepdims=True) #approximate softmax

tpr,fpr,roc_auc = ([[]]*N_classes for _ in range(3))
f,ax = plt.subplots()
#generate ROC data
for i in range(N_classes):
    fpr[i], tpr[i], _ = roc_curve(labels==i, preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    ax.plot(fpr[i],tpr[i])
plt.legend(['Class {:d}'.format(d) for d in range(N_classes)])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()