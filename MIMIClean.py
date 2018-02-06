#Cleaning MIMIC DB

from datetime import datetime
import pandas as pd
import numpy as np

def ICD9(adm_file="../ADMISSIONS.csv",diag_file="../DIAGNOSES_ICD.csv",ICD9_count=3,outfile=None):
    #This function gives back a df with the time of admissions of each patient
    #with the first admission as reference and their ICD9_count first diagnostics for each
    #admission. The ICD9 codes are cropped so that only the 3 first digits remain.
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
        data_s["ICD9_CODE_"+str(idx)]=data_s["ICD9_CODE_"+str(idx)].str[:3]
    #Add days granularity
    data_s["ELAPSED_DAYS"]=data_s["ELAPSED_TIME"].dt.days
    data_s.to_csv(outfile)
    return data_s
