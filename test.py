
import MIMICdev as dev
import MIMICtorch as mtorch

from MIMICtorch import EHRDataset
from MIMICtorch import model_train

import torch
import numpy as np

import matplotlib.pyplot as plt


#dummy data:
#pat=150
#cond=150
#K_train=2
#U_train=np.random.randn(pat,K_train,1) #Patient,latent_dim,time
#V_train=np.random.randn(K_train,cond) #latent_dim,condition
#X_prod=np.dot(U_train,V_train)
#X_prod=np.einsum('ijk,jl->ilk',U_train,V_train)
#X_prob=mtorch.sigmoid(X_prod)
#X_bin=np.random.binomial(1,X_prob)
#X_ids=np.asarray(np.where(~np.isnan(X_bin)))
#X_source=(X_ids,X_bin[tuple(X_ids)],X_bin.shape)

print("Sourcing Data ... ")
X_source=dev.matrix_creation() # Tuple with index and data of the matrix source.
print("Data Sourced")
print("Number of patients: "+str(X_source[2][0]))
print("Number of conditions: "+str(X_source[2][1]))
print("Number of time steps: "+str(X_source[2][2]))


print("Loading data ... ")
[Xtrain,Xval,Xtest]=mtorch.train_test(X_source,0.2,0.1)
ehr=EHRDataset(Xtrain)

mod=model_train(ehr,Xval,l_r=0.001,batch_size=200)
[U,V]=mod.run_train()

#train recap :
train_loss=mtorch.test_loss(Xtrain,U,V)
print("Overall train loss: "+str(train_loss))
#val recap :
val_loss=mtorch.test_loss(Xval,U,V)
print("Overall validation loss: "+str(val_loss))
#test:
test_loss=mtorch.test_loss(Xtest,U,V)
print("Overall test loss: "+str(test_loss))
print("Average probability of correct prediction: " +str(np.exp(-test_loss)))

print("Baseline :"+str(1-sum(Xtest[1])/len(Xtest[1])))

Unp=U.data.numpy()
Vnp=V.data.numpy()

#X_prob_inf=mtorch.sigmoid(np.einsum('ijk,jl->ilk',Unp,Vnp))[tuple(Xtest[0])]
#print('Mean difference of test probabilities : '+str(np.mean(np.abs(X_prob_inf-X_prob[tuple(Xtest[0])]))))


plt.plot(mod.Train_history,c="red",label="Training")
plt.plot(mod.Val_history,c="blue",label="Validation")
plt.title("Learning Curves")
plt.legend()
plt.show()
