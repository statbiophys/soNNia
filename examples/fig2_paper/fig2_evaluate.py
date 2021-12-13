import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('agg')
from sonia.evaluate_model import EvaluateModel
from sonnia.sonnia import SoNNia
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import olga.load_model as olga_load_model
import olga.generation_probability as generation_probability
import os
from matplotlib.lines import Line2D

#load test data
df=pd.read_csv('sampled_data/test_data.csv.gz')
to_evalutate=list(df[['amino_acid','v_gene','j_gene']].values)

#define model
qm = SoNNia(load_dir='selection_models/emerson_linear',custom_pgen_model='universal_model')
qm0 = SoNNia(load_dir='selection_models/emerson_NN',custom_pgen_model='universal_model')

# load Evaluate model
main_folder='universal_model'
params_file_name = os.path.join(main_folder,'model_params.txt')
marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')
genomic_data = olga_load_model.GenomicDataVDJ()
genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
generative_model = olga_load_model.GenerativeModelVDJ()
generative_model.load_and_process_igor_model(marginals_file_name)
pgen_model = generation_probability.GenerationProbabilityVDJ(generative_model, genomic_data)
ev=EvaluateModel(sonia_model=qm,custom_olga_model=pgen_model)
ev0=EvaluateModel(sonia_model=qm0,custom_olga_model=pgen_model)

#evaluate ppost/pgen
q,pgen,ppost=ev.evaluate_seqs(to_evalutate)
q_deep,pgen_deep,ppost_deep=ev0.evaluate_seqs(to_evalutate)
df['q']=q
df['pgen']=pgen
df['ppost']=ppost
df['q_deep']=q_deep
df['pgen_deep']=pgen_deep
df['ppost_deep']=ppost_deep
df.to_csv('sampled_data/test_data.csv.gz',compression='gzip',index=False)