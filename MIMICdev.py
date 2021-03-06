#Cleaning MIMIC DB

from datetime import datetime
import pandas as pd
import numpy as np
import progressbar

def ICD9(adm_file="/data/edebrouwer/MIMICs/ADMISSIONS.csv",diag_file="/data/edebrouwer/MIMICs/DIAGNOSES_ICD.csv",ICD9_count=3,outfile=None,digit_num=2):
    #This function gives back a df with the time of admissions of each patient
    #with the first admission as reference and their ICD9_count first diagnostics for each
    #admission. The ICD9 codes are cropped so that only the digit_num first digits remain.
    #ICD9 counts is the number of ICD9 codes we consider by admission.
    print("Reading from "+adm_file)
    adm=pd.read_csv(adm_file)
    df=adm.groupby("SUBJECT_ID")["HADM_ID"].nunique()
    subj_ids=list(df[df>1].index)
    adm_filtered=adm.loc[adm["SUBJECT_ID"].isin(subj_ids)]
    #Add the condition diagnostic in ICD9 codes.
    print("Reading from "+diag_file)
    diag=pd.read_csv(diag_file)
    data_m=adm_filtered
    #only select primary diagnosis :
    Diag_num=ICD9_count
    for idx in range(1,Diag_num+1):
        diag_idx=diag.loc[diag["SEQ_NUM"]==idx][["HADM_ID","ICD9_CODE"]]
        data_m=pd.merge(data_m,diag_idx,on="HADM_ID",how="left")
        data_m.rename(columns={"ICD9_CODE":"ICD9_CODE_"+str(idx)},inplace=True)

    data_m['ADMITTIME']=pd.to_datetime(data_m["ADMITTIME"], format='%Y-%m-%d %H:%M:%S')
    data_m['DISCHTIME']=pd.to_datetime(data_m["DISCHTIME"], format='%Y-%m-%d %H:%M:%S')

    admit_group=data_m.groupby("SUBJECT_ID",as_index=False)["ADMITTIME"].min()
    admit_group.rename(columns={"ADMITTIME":"REF_TIME"},inplace=True)
    data_m=pd.merge(data_m,admit_group,on="SUBJECT_ID")
    data_m["ELAPSED_TIME"]=data_m["ADMITTIME"]-data_m["REF_TIME"]

    #Add special rows for the death times
    data_m["DEATH"]=0
    dead_rows=data_m[data_m.DEATHTIME.notnull()][["SUBJECT_ID","DEATHTIME","REF_TIME"]]
    dead_rows['DEATHTIME']=pd.to_datetime(dead_rows["DEATHTIME"], format='%Y-%m-%d %H:%M:%S')
    dead_rows["ELAPSED_TIME"]=dead_rows["DEATHTIME"]-dead_rows["REF_TIME"]
    dead_rows["DEATH"]=1
    #Concatenate dfs
    data_death=pd.concat([data_m,dead_rows])

    #Select only the attributes we are interested in :
    col_selection=["SUBJECT_ID","ELAPSED_TIME","DEATH"]+list(data_m)[19:19+Diag_num]
    data_s=data_death[col_selection].sort_values(["SUBJECT_ID","ELAPSED_TIME"])
    #Clean the ICD9 to 3 digits
    for idx in range(1,Diag_num+1):
        data_s["ICD9_CODE_"+str(idx)]=data_s["ICD9_CODE_"+str(idx)].str[:digit_num]
    #Add days granularity
    data_s["ELAPSED_DAYS"]=data_s["ELAPSED_TIME"].dt.days

    print("Number of unique patients : "+str(len(data_s["SUBJECT_ID"].unique())))

    #Check number of unique ICD9 Codes
    ICD_serie=data_s["ICD9_CODE_1"]
    for idx in range(2,Diag_num+1):
        ICD_serie=ICD_serie.append(data_s["ICD9_CODE_"+str(idx)])
    print("Number of unique conditions : "+str(len(ICD_serie.unique())))

    data_s.to_csv(outfile)
    return data_s

def matrix_creation(ICD9_file="/data/edebrouwer/MIMICs/ICD9Clean.csv",granul=7,ICD9_count=3,ICD9_cap=5,Time_quantile=0.7):
    #Input file for the matrix creation
    #granul = The desired granularity in days.
    #Return a tuple with the index and values of the non empty elements of the matrix.
    dat=pd.read_csv(ICD9_file)
    Diag_num=ICD9_count

    #Give new indexes for subjects.
    old_ID=dat["SUBJECT_ID"].unique()
    n_dict=dict(zip(old_ID,range(0,len(old_ID))))
    dat["ID"]=dat["SUBJECT_ID"].map(n_dict)

    #Check number of unique ICD9 Codes and give new indexes.
    ICD_serie=dat["ICD9_CODE_1"]
    for idx in range(2,Diag_num+1):
        ICD_serie=ICD_serie.append(dat["ICD9_CODE_"+str(idx)])
    #Remove codes that  appear less than ICD9_cap times
    counts_ICD9=ICD_serie.value_counts()
    to_delete=counts_ICD9[counts_ICD9<ICD9_cap]
    ICD_serie.replace(to_delete.index,np.nan,inplace=True)

    unique_codes=np.unique(ICD_serie[~pd.isnull(ICD_serie)])

    #Attention : the index 0 is reserved for the NA in condition (usually corresponds to death.)
    ICD9_map=dict(zip(unique_codes,range(1,len(unique_codes)+1)))
    dat["ID"]=dat["SUBJECT_ID"].map(n_dict)
    condition_vect=[]
    for idx in range(1,Diag_num+1):
        dat["CONDITION_"+str(idx)]=dat["ICD9_CODE_"+str(idx)].map(ICD9_map).fillna(0).astype(int)
        condition_vect=condition_vect+["CONDITION_"+str(idx)]

    #Convert time to the required granularity (Here 5 days)
    dat["ELAPSED_5d"]=np.floor(dat["ELAPSED_DAYS"]/granul).astype(int)

    #Create the data matrix
    X=np.full((len(old_ID),len(unique_codes)+1,max(dat["ELAPSED_5d"])+1),fill_value=np.nan)

    #Fill the matrix:
    #We suppose that events that already occured and are not reported anymore in the future did not occur until the end of observation (fill with 0)
    for idx in range(0,len(old_ID)):
        l_u=list(set(dat[dat["ID"]==idx][condition_vect].values.flatten()))
        m_t=max(dat[dat["ID"]==idx]["ELAPSED_5d"])
        #X[idx,0,:m_t+1]=0 for death, see below.
        X[idx,l_u,:m_t+1]=0
    for cdx in range(1,Diag_num+1):
        X[dat["ID"],dat["CONDITION_"+str(cdx)],dat["ELAPSED_5d"]]=1
    #Fill DEATH (zeros until death)
    for idx in range(0,len(old_ID)):
        m_t=max(dat[dat["ID"]==idx]["ELAPSED_5d"])
        X[idx,0,:m_t+1]=0
        if max(dat[dat["ID"]==idx]["DEATH"])==1:
            X[idx,0,m_t]=1


    #Clean the time steps (remove the tail (0.9) of the distribution)
    X=clean_time(X,quantile=Time_quantile)

    #Clean the ICD9 codes that do not occur more than 2 times

    #Go for sparse representation.
    X_idx=np.asarray(np.where(~np.isnan(X)))
    X_dat=X[tuple(X_idx)]
    return (X_idx,X_dat,X.shape)

def clean_time(X,quantile=0.9):
    #remove the times for which only 1-quantile % of the patients reach this observation time.
    X_max=np.zeros(X.shape[0])
    for i in range(0,X.shape[0]):
        X_max[i]=max(np.where(~np.isnan(X[i,:,:]))[1])
    X_ordered_duration=np.sort(X_max)
    cap=int(X_ordered_duration[int(len(X_ordered_duration)*quantile)])
    return(X[:,:,:cap+1])

def clean_ICD9(X,cap=2):
    X_sum=np.nansum(X,axis=2)
    X_sum2=sum(X_sum)



def run_inference(X,K=2,sig2=0.2,iterT=20,lr=0.1):
    #latent vectors intialization
    U=0.1*np.random.randn(X.shape[0],K,X.shape[2]) #[patient x K x time]
    V=0.1*np.random.randn(K,X.shape[1]) #[K x conditions]
    bar= progressbar.ProgressBar()
    for loop in bar(range(0,iterT)):
        for t_idx in range(0,X.shape[2]):
            for u_idx in range(0,X.shape[0]):
                U[u_idx,:,t_idx]+=lr*grad_u(u_idx,t_idx,U,V,data_u=X[u_idx,:,t_idx],sig2=sig2)
            for v_idx in range(0,X.shape[1]):
                V[:,v_idx]+=lr*grad_v(t_idx,v_idx,U,V,data_v=X[:,v_idx,t_idx],sig2=sig2)
    return [U,V]

#function to return the gradient of the posterior with respect to U_i
#at t=t
def grad_u(u_idx,t_idx,U,V,data_u,sig2):
   # u_idx=1 #patient
   # t_idx=4 #time
   # v_idx=2 #condition
   # data_u=X[u_idx,:,t_idx]
    u=U[u_idx,:,t_idx]
    prod=np.exp(-np.dot(u,V))
    mask_u=data_u*2-1
    com_fact=(mask_u*(data_u*prod+(1-data_u)))/(1+prod)
    grad_1=np.nansum(V*com_fact,axis=1) #gradient of the vector u in each dimension.

    u_prev=U[u_idx,:,t_idx-1]
    if t_idx==(U.shape[2]-1):
        u_next=u
    else:
        u_next=U[u_idx,:,t_idx+1]
    grad_2=(u_prev+u_next-2*u)/sig2

    if (t_idx==0):
        grad_2=-u/sig2

    grad=grad_1+grad_2 #gradient for the u vector
    return grad

#function to return the gradient of the posterior with respect to V_j
def grad_v(t_idx,v_idx,U,V,data_v,sig2):
    #u_idx=1 #patient
    #t_idx=4 #time
    #v_idx=2 #condition
    #data_v=X[:,v_idx,t_idx]
    v=V[:,v_idx]
    prod=np.exp(-np.dot(U[:,:,t_idx],v))
    mask_v=data_v*2-1
    com_fact=(mask_v*(data_v*prod+(1-data_v)))/(1+prod)
    grad_1=np.nansum(U[:,:,t_idx].T*com_fact,axis=1) #gradient of the vector u in each dimension.

    grad_2=-v/sig2

    grad=grad_1+grad_2 #gradient for the u vector
    return grad

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dummy_data_gen(pat=100,cond=10,K=2,T=52,sig2_walk=0.2,method="logit"):
    #use random walk to generate the U's.
    U_train=np.zeros((pat,K,T)) #Patient,latent_dim,time
    for i in range(T):
        U_train[:,:,i]=np.sqrt(sig2_walk)*np.random.randn(pat,K)+U_train[:,:,i-1]
    V_train=np.sqrt(sig2_walk)*np.random.randn(K,cond)
    #V_train=np.ones((K,cond))

    X_prod=np.einsum('ijk,jl->ilk',U_train,V_train)
    X_prob=sigmoid(X_prod)
    X_bin=np.random.binomial(1,X_prob)

    if(method=="probit"):
        X_prob=X_prod+np.random.randn(pat,cond,T)
        X_bin=np.copy(X_prob)
        X_bin[X_bin<0]=0
        X_bin[X_bin>0]=1

    X_ids=np.asarray(np.where(~np.isnan(X_bin)))
    X_source=(X_ids,X_bin[tuple(X_ids)],X_bin.shape) #returns in the form of indices,values and total shape.
    return([X_source,X_bin,X_prob,U_train,V_train])
