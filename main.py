#
import MIMICdev as dev
import MIMICtorch as mtorch

from MIMICtorch import EHRDataset

import torch
import numpy as np

print("Sourcing Data ... ")
X_source=dev.matrix_creation() # Tuple with index and data of the matrix source.
print("Data Sourced")
print("Number of patients: "+str(X_source[2][0]))
print("Number of conditions: "+str(X_source[2][1]))
print("Number of time steps: "+str(X_source[2][2]))

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


#X_source=np.ones((10,10,10))

print("Loading data ... ")
[Xtrain,Xval,Xtest]=mtorch.train_test(X_source,0.2,0.1)
ehr=EHRDataset(Xtrain)
print("Number of data points : "+str(len(Xtrain[1])))
print("Data Loaded !")

#print("Pre-processing data")
#[X_train,X_mask,Xtest]=mtorch.pre_proc_data(X_source,prop=0.2)
#print("Pre-processing done !")

#training:
print("Training ... ")
[U,V]=mtorch.run_training(ehr,Xval,sig2=4,K=2,l_r=0.01,batch_size=100)
print("Done with training ! ")

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

Unp=U.data.numpy()
Vnp=V.data.numpy()

X_prob_inf=mtorch.sigmoid(np.einsum('ijk,jl->ilk',Unp,Vnp))[tuple(Xtest[0])]
print('Mean difference of test probabilities : '+str(np.mean(np.abs(X_prob_inf-X_prob[tuple(Xtest[0])]))))

np.save("Utorch",Unp)
np.save("Vtorch",Vnp)
np.save("Xtrain",Xtrain)
np.save("Xtest",Xtest)
