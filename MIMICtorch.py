#Pytorch functions

import numpy as np
import torch
from torch.autograd import Variable


def pre_proc_data(X_source,prop):
    [X,Xtest]=train_test(X_source,prop=prop)

    X_mask=np.ones(X.shape) #mask for position of training data
    nan_index=np.where(np.isnan(X))
    X_mask[nan_index]=0

    X=torch.from_numpy(X).type(torch.IntTensor) #conversion to pytorch tensor
    return[X,X_mask,Xtest]


def run_training(X,X_mask,sig2=4,K=2,l_r=0.01):

    U=Variable(0.1*torch.randn(X.shape[0],K,X.shape[2]),requires_grad=True)
    V=Variable(0.1*torch.randn(K,X.shape[1]),requires_grad=True)

    optimizer= torch.optim.Adam([U,V],lr=l_r)

    #convergence parameters
    delta=1
    prev=1

    data_idx=np.nonzero(X_mask)
    while(delta>1e-5):
        optimizer.zero_grad()
        total_loss=0
        for (i,j,t) in zip(*data_idx):
            y_pred=forward(U[i,:,t],V[:,j])
            loss=comp_loss(X[i,j,t],y_pred)
            total_loss+=loss
            #loss.backward()
            if ((y_pred!=y_pred).any()):
                print("Issue NAN")
                print(U)
                print(V)
        print("Loss is "+str(total_loss.data[0]))
        total_loss+=regul_loss(U,V,sig2)
        print("Loss with regul is "+str(total_loss.data[0]))
        delta=abs(prev-total_loss.data[0])
        prev=total_loss.data[0]
        total_loss.backward()
        optimizer.step()
    return([U,V,Xtest])

def forward(U,V):
    y=torch.sigmoid(torch.dot(U,V))
    #y=torch.dot(U[u_idx,:],V[:,v_idx])
    return(y)

def regul_loss(U,V,sig2):
    regul=torch.sum((V.pow(2))/sig2)+torch.sum((U[:,:,0].pow(2))/sig2)
    for t_idx in range(1,U.shape[2]):
        regul+=torch.sum(((U[:,:,t_idx]-U[:,:,t_idx-1]).pow(2))/sig2)
    return(regul)

def comp_loss(y_data,y_pred):
    loss=-((1-y_data)*torch.log(1-y_pred)+y_data*(torch.log(y_pred)))#+(1/len_v)*torch.sum((V[:,v_idx].pow(2))/sig2)+(1/len_u)*torch.sum((U[u_idx,:].pow(2))/sig2)
    #print(loss)
    #loss=(y_data-y_pred).pow(2)+torch.sum((V[:,2].pow(2))/2)
    return(loss)

def test_loss(Xtest,U,V):
    loss=0
    test_idx=np.where(~np.isnan(Xtest))
    for (i,j,t) in zip(*test_idx):
        y_data=Xtest[i,j,t]
        y_pred=forward(U[i,:,t],V[:,j])
        loss+=-((1-y_data)*torch.log(1-y_pred)+y_data*(torch.log(y_pred)))
    return(loss/len(test_idx[0]))




#utils:
def sigmoid(x):
    return 1/(1+np.exp(-x))

def gruyering(X,prop=0.1): #Returns a matrix with a proportion of its entries = 0
    zero_num=int(prop*np.prod(X.shape))
    idx1=np.random.choice(range(0,X.shape[0]),zero_num)
    idx2=np.random.choice(range(0,X.shape[1]),zero_num)
    idx3=np.random.choice(range(0,X.shape[2]),zero_num)
    X[idx1,idx2,idx3]=0
    return(X)

def train_test(X,prop): #Divide the matrix in train and test set. Non values are set to nan.
    valid_index=np.asarray(np.where(~np.isnan(X)))
    test_num=int(prop*len(valid_index[0]))
    test_idx=np.random.choice(range(0,len(valid_index[0])),test_num)
    test_ids=tuple(valid_index[:,test_idx])

    Xtest=np.empty(X.shape)
    Xtest[:]=np.nan
    Xtest[test_ids]=X[test_ids]
    X[test_ids]=np.nan
    return([X,Xtest])
