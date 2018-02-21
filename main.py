#
import MIMICdev as dev
import MIMICtorch as mtorch

import torch
import numpy as np

X_source=dev.matrix_creation()

[X_train,X_mask,Xtest]=mtorch.pre_proc_data(X_source,prop=0.2)

#training:
[U,V]=mtorch.run_training(X_train,X_mask,sig2=4,K=2,l_r=0.01)

#test:
test_loss=mtorch.test_loss(U,V,Xtest)
print(test_loss)
print(np.exp(-test_loss))