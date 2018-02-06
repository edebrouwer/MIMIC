#Cleaning MIMIC DB

from datetime import datetime
import pandas as pd
import numpy as np

def ICD9(adm_file="../ADMISSIONS.csv",diag_file="../DIAGNOSES_ICD.csv",ICD9_count=3):
    #This function gives back a df with the time of admissions of each patient
    #with the first admission as reference and their ICD9_count first diagnostics for each
    #admission. The ICD9 codes are cropped so that only the 3 first digits remain.
    print(adm_file)
    adm=pd.read_csv(adm_file)
    df=adm.groupby("SUBJECT_ID")["HADM_ID"].nunique()
    subj_ids=list(df[df>1].index)
    adm_filtered=adm.loc[adm["SUBJECT_ID"].isin(subj_ids)]
    #Add the condition diagnostic in ICD9 codes.
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
    #Select only the attributes we are interested in :
    col_selection=["SUBJECT_ID","ELAPSED_TIME"]+list(data_m)[19:19+Diag_num]
    data_s=data_m[col_selection]
    #Clean the ICD9 to 3 digits
    for idx in range(1,Diag_num+1):
        data_s["ICD9_CODE_"+str(idx)]=data_s["ICD9_CODE_"+str(idx)].str[:3]
    return data_s
