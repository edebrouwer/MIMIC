#Pytorch functions

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import ExponentialLR
import time


def pre_proc_data(X_source,prop):

    X_idx=np.asarray(np.where(~np.isnan(X))) #index of non nan datas
    X_vals=X[X_idx]

    [Xtrain_idx,Xtrain_vals,Xtest_idx,Xtest_vals]=train_test(X_idx,X_vals,prop=prop)

    X_mask=np.ones(X.shape) #mask for position of training data
    nan_index=np.where(np.isnan(X))
    X_mask[nan_index]=0

    X=torch.from_numpy(X).type(torch.IntTensor) #conversion to pytorch tensor
    return[X,X_mask,Xtest]


def run_training(ehr_data,Xval,sig2=4,K=2,l_r=0.01,**opt_args):

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
    Train_history=np.array([])
    Val_history=np.array([])
    delta_val=1
    delta_train=1
    prev_val=1
    prev_train=1
    val_loss=0
    agg_loss=0
    tol=1e-5
    check_freq=20 #every x batches, we compute training and validation loss.
    for epochs in range(0,100):
        #for (i,j,t) in zip(*data_idx):
        print("Epoch :"+str(epochs))
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




            total_loss/=len(sample['data'])
            regul=regul_loss(U,V,sig2)/len(ehr_data) #A verifier !!!
            total_loss+=regul # A VERIFIER
            agg_loss+=total_loss.data[0] #Used for convergence check

            total_loss.backward()
            optimizer.step()

            if ((i_batch+1) % 20 == 0):
                val_loss=test_loss(Xval,U,V)+regul_loss(U,V,sig2).data[0]/len(Xval[1])
                Val_history=np.append(Val_history,val_loss)
                agg_loss/=20
                Train_history=np.append(Train_history,val_loss)
                print("Validation Loss : "+str(val_loss))
                print("Training Loss : "+str(agg_loss))
                delta_val=abs(prev_val-val_loss)
                prev_val=val_loss
                delta_train=abs(prev_train-agg_loss)
                prev_train=agg_loss
                agg_loss=0
            if ( delta_train<tol or delta_val<tol):
                print("BREAK")
                return([U,V])

        #agg_loss=agg_loss/len(ehr_data)+regul
        #print(agg_loss.data[0])

    return([U,V])

def forward(U,V):
    y=torch.sigmoid(torch.dot(U,V))
    #y=torch.dot(U[u_idx,:],V[:,v_idx])
    return(y)

def forward_normal(U,V):
    y=torch.dot(U,V)
    return(y)

def regul_loss(U,V,sig2):
    regul=torch.sum((V.pow(2))/sig2)+torch.sum((U[:,:,0].pow(2))/sig2)
    for t_idx in range(1,U.shape[2]):
        regul+=torch.sum(((U[:,:,t_idx]-U[:,:,t_idx-1]).pow(2))/sig2)
    return(regul)

#def regul_loss(U,V,i,j,t,sig2):
#    regul=torch.sum((V[:,j].pow(2))/sig2) #For V
#    #Now for U. Two special cases : when t=0 and when t=T
#    if t==0:
#        regul+=torch.sum((U[i,:,t].pow(2))/sig2)
#    elif t==(U.shape[2]-1):
#        regul+=torch.sum(((U[i,:,t]-U[i,:,t-1]).pow(2))/sig2)
#    else:
#        regul+=torch.sum(((U[i,:,t]-U[i,:,t-1]).pow(2))/sig2)+torch.sum(((U[i,:,t+1]-U[i,:,t]).pow(2))/sig2)
#    return(regul)

def comp_loss(y_data,y_pred):
    print("OUTSIDE")
    loss=-((1-y_data)*torch.log(1-y_pred)+y_data*(torch.log(y_pred)))#+(1/len_v)*torch.sum((V[:,v_idx].pow(2))/sig2)+(1/len_u)*torch.sum((U[u_idx,:].pow(2))/sig2)
    #print(loss)
    #loss=(y_data-y_pred).pow(2)+torch.sum((V[:,2].pow(2))/2)
    return(loss)

def test_loss(Xtest,U,V): #returns a float( not a Tensor !!!)
    loss=0
    test_idx=Xtest[0]
    for id_y,y in enumerate(Xtest[1]):
        y_data=y
        i=test_idx[0,id_y]
        j=test_idx[1,id_y]
        t=test_idx[2,id_y]
        y_pred=forward(U[i,:,t],V[:,j]).data[0]

        #This is a dirty hack, look for better solution !!
        if (y_pred==1):
            print("y_pred is one")
            y_pred=0.99999
        if (y_pred==0):
            print("y_pred is zero")
            y_pred=0.00001

        loss+=-((1-y_data)*np.log(1-y_pred)+y_data*(np.log(y_pred)))
    return(loss/len(test_idx[0]))

def test_loss_normal(Xtest,U,V,sig2=4): #returns a float( not a Tensor !!!)
    loss=0
    test_idx=Xtest[0]
    for id_y,y in enumerate(Xtest[1]):
        y_data=y
        i=test_idx[0,id_y]
        j=test_idx[1,id_y]
        t=test_idx[2,id_y]
        y_pred=forward_normal(U[i,:,t],V[:,j]).data[0]
        loss+=((y_data-y_pred)/sig2)**2
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

def train_test(X,prop_val,prop_test): #Divide the matrix in train validation and test set. Non values are set to nan.

    #valid_index=np.asarray(np.where(~np.isnan(X)))
    train_samps=len(X[0][0]) #number of non nan values
    test_num=int(prop_test*train_samps) #Number of test values
    val_num=int(prop_val*train_samps)
    permut=np.random.permutation(len(X[0][0]))

    test_idx=permut[:test_num]
    val_idx=permut[test_num:test_num+val_num]
    train_idx=permut[test_num+val_num:]

    test_ids=X[0][:,test_idx]
    val_ids=X[0][:,val_idx]
    train_ids=X[0][:,train_idx]

    Xtest=X[1][test_idx]
    Xtrain=X[1][train_idx]
    Xval=X[1][val_idx]

    return([(train_ids,Xtrain,X[2]),(val_ids,Xval,X[2]),(test_ids,Xtest,X[2])])

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

class EHRDataset3(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X_source, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.idx=torch.from_numpy(X_source[0])
        self.data=torch.from_numpy(X_source[1])
        self.shape=X_source[2]

    def __len__(self):
        return len(self.idx[0])


    def __getitem__(self, idx):

        #sample={'data': self.data[idx],'i':self.idx[0][idx],'j':self.idx[1][idx],'t':self.idx[2][idx]}

        return self.idx[:,idx],self.data[idx]

class model_train():

    def __init__(self,ehr_data,Xval,sig2_prior=4,sig2_lik=2,K=2,l_r=0.01,epochs_num=100,check_freq=40,learning_decay=0.99,regul_inv=0.00001,l_kernel=1,**opt_args):
        self.ehr=ehr_data
        if ('batch_size' in opt_args):
            batch_size=opt_args['batch_size']
            self.check_freq=check_freq #every x batches, we compute training and validation loss.
            self.learning_decay=learning_decay
            print("Batch size ="+str(batch_size))
        else:
            batch_size=len(ehr_data) #by default the batch size is set to the full length of the data.
            self.check_freq=1
            self.learning_decay=1
            print("Full batch")


        self.ehr_loader=DataLoader(ehr_data,batch_size=batch_size,shuffle=True,num_workers=0)

        self.T=ehr_data.shape[2] #Number of time steps

        self.U=Variable(0.1*torch.randn(ehr_data.shape[0],K,self.T),requires_grad=True)
        self.V=Variable(0.1*torch.randn(K,ehr_data.shape[1]),requires_grad=True)
        #self.V=Variable(torch.ones(K,ehr_data.shape[1]))

        self.Xval=Xval

        self.epochs_num=epochs_num
        self.l_r=l_r

        self.sig2_prior=Variable(torch.from_numpy(np.array([sig2_prior])).type(torch.FloatTensor),requires_grad=True)
        self.sig2_lik=sig2_lik

        self.regul_inv=regul_inv # regularization for the inversion of the kernel matrix.

        #Stores the train and validation error over the training process.
        self.Train_history=np.array([])
        self.Val_history=np.array([])

        #GP inverse Kernel
        if ('kernel_type' in opt_args):
            self.kernel_type=opt_args['kernel_type']
            x_samp=np.linspace(0,self.T-1,self.T)
            self.SexpKernel=np.exp(-(np.array([x_samp]*self.T)-np.expand_dims(x_samp.T,axis=1))**2/(2*l_kernel**2))
            self.inv_Kernel=Variable(torch.from_numpy(np.linalg.inv(self.SexpKernel+self.regul_inv*np.identity(self.T))).type(torch.FloatTensor))#/sig2_prior
        else:
            self.kernel_type="random-walk" #by default.
            self.SexpKernel=0
            self.inv_Kernel=0
        print("Kernel used is : "+self.kernel_type)

        #Convergence parameters.
        self.delta_val=1
        self.delta_train=1
        self.prev_val=1e5
        self.prev_train=1e5
        self.val_loss=0
        self.agg_loss=0
        self.tol=1e-6


    def run_train(self):
        optimizer= torch.optim.Adam([self.U,self.V,self.sig2_prior],lr=self.l_r)
        #optimizer= torch.optim.Adam([self.U],lr=self.l_r)
        scheduler_lr=ExponentialLR(optimizer, gamma=self.learning_decay) #Exponential Decay

        if (self.kernel_type=='square-exp'):
            regul_loss_fun=self.regul_loss_GP
        elif (self.kernel_type=='random-walk'):
            regul_loss_fun=self.regul_loss
        else:
            print("Unknown kernel function")
            return([self.U,self.V])

        try:
            for epochs in range(0,self.epochs_num):
                scheduler_lr.step()
                #for (i,j,t) in zip(*data_idx):
                self.agg_loss=0
                print("Epoch :"+str(epochs)+" number of data samples ="+str(len(self.ehr)))

                i_batch=0
                T1=time.time()
                for sample in self.ehr_loader:
                    print ("Loading one batch is " + str(time.time()-T1))
                    T0=time.time()
                    optimizer.zero_grad()
                    total_loss=0
                    #for data_sample,i,j,t in zip(sample['data'],sample['i'],sample['j'],sample['t']): #For  first EHRDataLoader

                    for data_sample,i,j,t in zip(sample[1],sample[0][:,0],sample[0][:,1],sample[0][:,2]):
                        y_pred=self.forward(self.U[i,:,t],self.V[:,j])
                        loss=self.comp_loss(data_sample,y_pred)
                        total_loss+=loss

                    #total_loss/=len(sample['data']) #For first EHRDataLoader
                    total_loss/=sample[1].shape[0]

                    regul=regul_loss_fun(self.U,self.V,self.sig2_prior)/len(self.ehr) #A verifier !!!
                    total_loss+=regul # A VERIFIER
                    self.agg_loss+=total_loss.data[0] #Used for convergence check

                    total_loss.backward()
                    optimizer.step()

                    if ((i_batch+1) % self.check_freq == 0):
                        self.val_loss=self.test_loss(self.Xval,self.U,self.V)+regul_loss_fun(self.U,self.V,self.sig2_prior).data[0]/len(self.ehr)
                        self.Val_history=np.append(self.Val_history,self.val_loss)
                        self.agg_loss/=self.check_freq
                        self.Train_history=np.append(self.Train_history,self.agg_loss)
                        print("Validation Loss : "+str(self.val_loss))
                        print("Training Loss : "+str(self.agg_loss))
                        self.delta_val=self.prev_val-self.val_loss
                        self.prev_val=self.val_loss
                        self.delta_train=abs(self.prev_train-self.agg_loss)
                        self.prev_train=self.agg_loss
                        self.agg_loss=0
                    if ( self.delta_train<self.tol or self.delta_val<self.tol):
                        print("BREAK")
                        return([self.U,self.V])
                    print ("Processing One batch is " + str(time.time()-T0))
                    i_batch+=1
                    T1=time.time()
        except KeyboardInterrupt:
            print("Training Stopped by user")
            return([self.U,self.V])

            #agg_loss=agg_loss/len(ehr_data)+regul
            #print(agg_loss.data[0])
        print("No convergence !")
        return([self.U,self.V])

    def forward(self,U,V):
        y=torch.sigmoid(torch.dot(U,V))
        #y=torch.dot(U[u_idx,:],V[:,v_idx])
        return(y)

    def forward_normal(self,U,V):
        y=torch.dot(U,V)
        return(y)

    def comp_loss(self,y_data,y_pred):
        #This is a dirty hack, look for better solution !!
        if (y_pred.data[0]==1):
            print("y_pred is one")
            y_pred.data[0]=0.99999
        if (y_pred.data[0]==0):
            print("y_pred is zero")
            y_pred.data[0]=0.00001

        loss=-((1-y_data)*torch.log(1-y_pred)+y_data*(torch.log(y_pred)))

        return(loss)

    def comp_loss_normal(self,y_data,y_pred):
        loss=((y_data-y_pred)/(2*self.sig2_lik)).pow(2)
        return(loss)

    def regul_loss_GP(self,U,V,sig2):
        K=U.shape[1]
        regul=0.5*torch.sum((V.pow(2))/sig2)+torch.sum((U[:,:,0].pow(2))/sig2)+0.1*sig2
        for p_idx in range(1,U.shape[0]):
            regul+=0.5*torch.sum(torch.mm(U[p_idx,:,:],torch.mm(self.inv_Kernel,U[p_idx,:,:].t()))[range(K),range(K)])/sig2
        return(regul)


    def regul_loss(self,U,V,sig2):
        regul=torch.sum((V.pow(2))/(2*sig2))+torch.sum((U[:,:,0].pow(2))/(2*sig2))
        for t_idx in range(1,U.shape[2]):
            regul+=torch.sum(((U[:,:,t_idx]-U[:,:,t_idx-1]).pow(2))/(2*sig2))
        return(regul)



    def test_loss(self,Xtest,U,V): #returns a float( not a Tensor !!!)
        loss=0
        test_idx=Xtest[0]
        for id_y,y in enumerate(Xtest[1]):
            y_data=y
            i=test_idx[0,id_y]
            j=test_idx[1,id_y]
            t=test_idx[2,id_y]
            y_pred=self.forward(U[i,:,t],V[:,j]).data[0]

            #This is a dirty hack, look for better solution !!
            if (y_pred==1):
                print("y_pred is one")
                y_pred=0.99999
            if (y_pred==0):
                print("y_pred is zero")
                y_pred=0.00001

            loss+=-((1-y_data)*np.log(1-y_pred)+y_data*(np.log(y_pred)))
        return(loss/len(test_idx[0]))


    def test_loss_normal(self,Xtest,U,V): #returns a float( not a Tensor !!!)
        loss=0
        test_idx=Xtest[0]
        for id_y,y in enumerate(Xtest[1]):
            y_data=y
            i=test_idx[0,id_y]
            j=test_idx[1,id_y]
            t=test_idx[2,id_y]
            y_pred=self.forward_normal(U[i,:,t],V[:,j]).data[0]
            loss+=((y_data-y_pred)/self.sig2_lik)**2
        return(loss/len(test_idx[0]))

    def run_train_normal(self):
        optimizer= torch.optim.Adam([self.U,self.V],lr=self.l_r)
        for epochs in range(0,self.epochs_num):
            #for (i,j,t) in zip(*data_idx):
            self.agg_loss=0
            print("Epoch :"+str(epochs))
            for i_batch, sample in enumerate(self.ehr_loader):
                optimizer.zero_grad()
                total_loss=0
                for data_sample,i,j,t in zip(sample['data'],sample['i'],sample['j'],sample['t']):
                    y_pred=self.forward_normal(self.U[i,:,t],self.V[:,j])
                    loss=self.comp_loss_normal(data_sample,y_pred)
                    total_loss+=loss


                total_loss/=len(sample['data'])
                regul=self.regul_loss(self.U,self.V,self.sig2_prior)/len(self.ehr) #A verifier !!!
                total_loss+=regul # A VERIFIER
                self.agg_loss+=total_loss.data[0] #Used for convergence check

                total_loss.backward()
                optimizer.step()

                if ((i_batch+1) % self.check_freq == 0):
                    self.val_loss=self.test_loss_normal(self.Xval,self.U,self.V)+regul_loss(self.U,self.V,self.sig2_prior).data[0]/len(self.ehr)
                    self.Val_history=np.append(self.Val_history,self.val_loss)
                    self.agg_loss/=self.check_freq
                    self.Train_history=np.append(self.Train_history,self.agg_loss)
                    print("Validation Loss : "+str(self.val_loss))
                    print("Training Loss : "+str(self.agg_loss))
                    self.delta_val=abs(self.prev_val-self.val_loss)
                    self.prev_val=self.val_loss
                    self.delta_train=abs(self.prev_train-self.agg_loss)
                    self.prev_train=self.agg_loss
                    self.agg_loss=0
                if ( self.delta_train<self.tol or self.delta_val<self.tol): #Break if training loss or validation loss stop varying or if validation loss increases !
                    print("BREAK")
                    return([self.U,self.V])

            #agg_loss=agg_loss/len(ehr_data)+regul
            #print(agg_loss.data[0])
        print("No convergence !")
        return([self.U,self.V])
