#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from sonnia.processing import Processing
from sonnia.utils import run_terminal,sample_olga
from scipy import stats

#directory location of emerson files
emerson_dict='emerson-nat/' 

# initialize preprocessing pipeline
processing_pipeline=Processing(custom_model_folder='universal_model',read_thresh=1)

#list files
emerson_data=[i for i in os.listdir(emerson_dict) if i[:3]=='HIP' or i[:3]=='Kec']

# load bad sequences: defined in deWitt: https://elifesciences.org/articles/38358.pdf
bad_seqs=pd.read_csv('bad_sequences.tds',sep='\t')['CDR3-amino-acids'].values

# prepare for parallelization
def process_emerson(file):
    print (file.split('/')[-1].split('.')[0])
    data=pd.read_csv(emerson_dict+file,sep='\t') # load
    data['reads']=data.frequency/data.frequency.min() # get number of reads
    keep=data.amino_acid.apply(lambda x: not x in bad_seqs ) # get rid of bad seqs
    data=data.loc[keep].reset_index(drop=True) 
    processed_data=processing_pipeline.filter_dataframe(data,apply_selection=False) # filter
    df=processed_data[['rearrangement','amino_acid','v_gene','j_gene']].loc[processed_data.selection] # apply selection
    df.to_csv('emerson_processed/'+file.split('/')[-1],index=False,compression='gzip') # save to file
    data=processed_data.loc[np.logical_and(processed_data.selection_genes,np.logical_not(processed_data.selection_productive))] # select unproductive seqs but with good genes (no pseudogenes)
    if len(data)>int(1e3):data=data.sample(n=int(1e3)).reset_index(drop=True)
    return data.rearrangement.values

# run parallelized filtering
processes=mp.cpu_count()
pool = mp.Pool(processes=processes)
f=pool.map(process_emerson, emerson_data)
pool.close()
out_of_frame=np.concatenate(f)
np.random.shuffle(out_of_frame)
np.savetxt('sampled_data/out_frame_seqs.txt',out_of_frame,fmt='%s') # save out of frame sequences to infer a common pgen model

# pool everything together
emerson_data=[i for i in os.listdir('emerson_processed') if i[:3]=='HIP' or i[:3]=='Kec']
print ('sample emerson')
emerson_directory='emerson_processed/'
np.random.shuffle(emerson_data)

# prepare for parallelization
def sample_emerson(file):
    d=pd.read_csv(emerson_directory+file).drop_duplicates()[['amino_acid','v_gene','j_gene']] # get rid of duplicates
    if len(d)>int(1e4):
        return d
    else: return []
    
processes=mp.cpu_count()
pool = mp.Pool(processes=processes)
dfs=pool.map(sample_emerson, emerson_data)
pool.close()
dfs=[df for df in dfs if len(df)>0]
print(len(dfs), 'repertoires pass minimum requirements')

# concatenate all repertoires
df_concatenated=pd.concat(dfs).sample(frac=1).reset_index(drop=True)
dfs=0

# get rid of TRBJ2-5 (badly annotated by the experimentalists)
df_concatenated=df_concatenated.loc[df_concatenated.j_gene!='TRBJ2-5']

#split train test
train=df_concatenated.sample(frac=0.5)
test=df_concatenated.loc[df_concatenated.index.difference(train.index), ]
print (len(train),len(test))

#group and compute frequencies
keys=['amino_acid','v_gene','j_gene']
test=test[keys].groupby(test[keys].columns.tolist()).size().reset_index().rename(columns={0:'freq'})
train=train[keys].groupby(train[keys].columns.tolist()).size().reset_index().rename(columns={0:'freq'})
print(len(test),len(train))

#normalise
test['log_freq']=np.log(test.freq/np.sum(test.freq))
train['log_freq']=np.log(train.freq/np.sum(train.freq))
print (len(test),len(train))

#shuffle
test=test.sample(frac=1).reset_index(drop=True)
train=train.sample(frac=1).reset_index(drop=True)

#sample data uniformly from train and test (otherwise too much data)
sampled_train=train[keys].reset_index(drop=True).sample(n=int(2e7),weights=np.exp(train.log_freq),replace=True)
sampled_test=test.reset_index(drop=True).sample(n=int(5e5),weights=np.exp(test.log_freq),replace=True)
sampled_test=sampled_test.loc[sampled_test.freq>1]
print(len(sampled_test),len(sampled_train))

# save data
sampled_train.reset_index(drop=True)[keys].to_csv('sampled_data/train_data.csv.gz',index=False,compression='gzip')
sampled_test.reset_index(drop=True).to_csv('sampled_data/test_data.csv.gz',index=False,compression='gzip')

# sample sequences from pgen
main_folder='universal_model/'
gen=sample_olga(int(2e7),custom_model_folder=main_folder)
pd.DataFrame(gen,columns=['amino_acid','v_gene','j_gene']).to_csv('sampled_data/generated_sequences.csv.gz',index=False,compression='gzip')  