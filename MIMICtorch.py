#Pytorch functions

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader



def pre_proc_data(X_source,prop):

    X_idx=np.asarray(np.where(~np.isnan(X))) #index of non nan datas
    X_vals=X[X_idx]

    [Xtrain_idx,Xtrain_vals,Xtest_idx,Xtest_vals]=train_test(X_idx,X_vals,prop=prop)

    X_mask=np.ones(X.shape) #mask for position of training data
    nan_index=np.where(np.isnan(X))
    X_mask[nan_index]=0

    X=torch.from_numpy(X).type(torch.IntTensor) #conversion to pytorch tensor
    return[X,X_mask,Xtest]


def run_training(ehr_data,sig2=4,K=2,l_r=0.01,**opt_args):

    if ('batch_size' in opt_args):
        batch_size=opt_args['batch_size']
        print("Batch size ="+str(batch_size))
    else:
        batch_size=len(ehr_data) #by default the batch size is set to the full length of the data.
        print("Full batch")

    ehr_loader=DataLoader(ehr_data,batch_size=batch_size)

    U=Variable(0.1*torch.randn(ehr_data.shape[0],K,ehr_data.shape[2]),requires_grad=True)
    V=Variable(0.1*torch.randn(K,ehr_data.shape[1]),requires_grad=True)

    optimizer= torch.optim.Adam([U,V],lr=l_r)

    #convergence parameters
    delta=1
    prev=1
    while(delta>1e-5):
        #for (i,j,t) in zip(*data_idx):
        for i_batch, sample in enumerate(ehr_loader):
            optimizer.zero_grad()
            total_loss=0
            for data_sample,i,j,t in zip(sample['data'],sample['i'],sample['j'],sample['t']):
                y_pred=forward(U[i,:,t],V[:,j])
                loss=comp_loss(data_sample,y_pred)
                total_loss+=loss
                #Some check if gradient does'nt go to a singularity.
                if ((y_pred!=y_pred).any()):
                    print("Issue NAN")
                    print(U)
                    print(V)
                total_loss+=regul_loss(U,V,i,j,t,sig2)
            print("Loss is "+str(total_loss.data[0]))
            #total_loss+=regul_loss(U,V,sig2)
            #print("Loss with regul is "+str(total_loss.data[0]))
            delta=abs(prev-total_loss.data[0])
            prev=total_loss.data[0]
            total_loss.backward()
            optimizer.step()
    return([U,V])

def forward(U,V):
    y=torch.sigmoid(torch.dot(U,V))
    #y=torch.dot(U[u_idx,:],V[:,v_idx])
    return(y)

#def regul_loss(U,V,sig2):
#    regul=torch.sum((V.pow(2))/sig2)+torch.sum((U[:,:,0].pow(2))/sig2)
#    for t_idx in range(1,U.shape[2]):
#        regul+=torch.sum(((U[:,:,t_idx]-U[:,:,t_idx-1]).pow(2))/sig2)
#    return(regul)

def regul_loss(U,V,i,j,t,sig2):
    regul=torch.sum((V[:,j].pow(2))/sig2) #For V
    #Now for U. Two special cases : when t=0 and when t=T
    if t==0:
        regul+=torch.sum((U[i,:,t].pow(2))/sig2)
    elif t==(U.shape[2]-1):
        regul+=torch.sum(((U[i,:,t]-U[i,:,t-1]).pow(2))/sig2)
    else:
        regul+=torch.sum(((U[i,:,t]-U[i,:,t-1]).pow(2))/sig2)+torch.sum(((U[i,:,t+1]-U[i,:,t]).pow(2))/sig2)
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

    #valid_index=np.asarray(np.where(~np.isnan(X)))
    train_samps=len(X[0][0]) #number of non nan values
    test_num=int(prop*train_samps) #Number of test values
    permut=np.random.permutation(len(X[0][0]))

    test_idx=permut[:test_num]
    train_idx=permut[test_num:]

    test_ids=X[0][:,test_idx]
    train_ids=X[0][:,train_idx]

    Xtest=X[1][test_idx]
    Xtrain=X[1][train_idx]

    return([(train_ids,Xtrain,X[2]),(test_ids,Xtest,X[2])])

class EHRDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X_source, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.idx=X_source[0]
        self.data=X_source[1]
        self.shape=X_source[2]

    def __len__(self):
        return len(self.idx[0])


    def __getitem__(self, idx):

        sample={'data': self.data[idx],'i':self.idx[0][idx],'j':self.idx[1][idx],'t':self.idx[2][idx]}

        return sample
