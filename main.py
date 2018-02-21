#
import MIMICdev as dev
import MIMICtorch as mtorch

import torch
import numpy as np

X_source=dev.matrix_creation()
#X_source=np.ones((10,10,10))

[X_train,X_mask,Xtest]=mtorch.pre_proc_data(X_source,prop=0.2)

#training:
[U,V]=mtorch.run_training(X_train,X_mask,sig2=4,K=2,l_r=0.01)

#test:
test_loss=mtorch.test_loss(Xtest,U,V)
print(test_loss)
print(np.exp(-test_loss))

Unp=U.data.numpy()
Vnp=V.data.numpy()

np.save("Utorch",Unp)
np.save("Vtorch",Vnp)
np.save("Xtrain",X_train.numpy())
np.save("Xtest",Xtest)
