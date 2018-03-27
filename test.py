
import MIMICdev as dev
import MIMICtorch as mtorch

from MIMICtorch import EHRDataset
from MIMICtorch import model_train

import torch
import numpy as np

import matplotlib.pyplot as plt


#dummy data creation :
[X_source,X_prob,U_train,V_train]=dev.dummy_data_gen(pat=20,cond=20,K=1,T=20,sig2_walk=0.2)

print("Sourcing Data ... ")
X_source=dev.matrix_creation() # Tuple with index and data of the matrix source.
print("Data Sourced")
print("Number of patients: "+str(X_source[2][0]))
print("Number of conditions: "+str(X_source[2][1]))
print("Number of time steps: "+str(X_source[2][2]))


print("Loading data ... ")
[Xtrain,Xval,Xtest]=mtorch.train_test(X_source,0.2,0.1)
ehr=EHRDataset(Xtrain)
print("Number of data points : "+str(len(Xtrain[1])))
print("Data loaded !")

mod=model_train(ehr,Xval,l_r=0.005,epochs_num=500,batch_size=100,sig2_prior=2,K=1,check_freq=10,l_kernel=3,kernel_type="square-exp")
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


np.save("Utorch",Unp)
np.save("Vtorch",Vnp)

np.save("Utrain",U_train)
np.save("Vtrain",V_train)

np.save("Xtrain",Xtrain)
np.save("Xtest",Xtest)

np.save("Xtrain_hist",mod.Train_history)
np.save("Xval_hist",mod.Val_history)


X_prob_inf=mtorch.sigmoid(np.einsum('ijk,jl->ilk',Unp,Vnp))[tuple(Xtest[0])]
print('Mean difference of test probabilities : '+str(np.mean(np.abs(X_prob_inf-X_prob[tuple(Xtest[0])]))))
print('Baseline : '+str(np.sqrt(np.var(X_prob[tuple(Xtest[0])]))))

plt.plot(mod.Train_history,c="red",label="Training")
plt.plot(mod.Val_history,c="blue",label="Validation")
plt.title("Learning Curves")
plt.legend()
plt.show()
